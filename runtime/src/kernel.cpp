#include "psprecore/runtime/kernel.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <stdexcept>

#include "core/foundation/containers/small_vector.h"
#include "core/foundation/core/foundation.h"
#include "core/foundation/core/time.h"
#include "core/foundation/strings/utf8_string.h"

namespace psprecore::rt {

namespace {

thread_local Thread* tl_current = nullptr;  // null on the idle (host main) thread and in interrupts

// Takes a baton, spinning briefly before sleeping: guest threads hand the baton back and forth many times per
// vblank, and waking a sleeping host thread costs tens of microseconds (more on phones), while the baton usually
// comes back within that time. Unthrottled (fast-forwarding) the handoffs are what limits the speed, so the spin is
// longer there: 0.1 ms per vblank on a phone instead of 5 ms without spinning; in real time a short spin saves the
// CPU (and battery) the waiting threads would burn. Adaptive: a baton whose recent waits outlasted the spin (a thread
// that sleeps until the next vblank, say) spins only briefly, so long waits do not burn the CPU either.
std::atomic<std::int64_t> g_baton_spin_us{50};

void take_baton(nx::semaphore& baton, std::uint8_t& misses) {
    std::int64_t spin_us = g_baton_spin_us.load(std::memory_order_relaxed);
    if (misses >= 2 && spin_us <= 50) spin_us = 3;  // real time only: fast-forwarding wants every handoff fast
    const nx::timestamp until = nx::timestamp::now() + nx::duration::from_micros(spin_us);
    do {
        for (int i = 0; i < 256; ++i)
            if (baton.try_acquire()) {
                misses = 0;
                return;
            }
    } while (nx::timestamp::now() < until);
    if (misses < 255) ++misses;
    baton.acquire();
}
Kernel* g_kernel = nullptr;

constexpr std::uint64_t kVblankPeriodUs = 16683;  // 59.94 Hz
constexpr int kVblankInterrupt = 30;

}  // namespace

Kernel& kernel() {
    if (!g_kernel) throw std::logic_error("kernel not set");
    return *g_kernel;
}
void set_kernel(Kernel* k) { g_kernel = k; }

Kernel::Kernel(GuestMemory& memory)
    : memory_(memory), user_heap_(kUserRam, kRam + kRamSize), kernel_heap_(kRam + 0x10000, 0x08400000) {
    interrupt_stack_ = kernel_heap_.alloc(0x4000, 16, true);
}

Kernel::~Kernel() {
    for (auto& [_, t] : threads_)
        if (t->host.joinable()) t->host.detach();  // threads still blocked at shutdown
}

Thread* Kernel::thread(int id) {
    const auto it = threads_.find(id);
    return it == threads_.end() ? nullptr : it->second.get();
}

Thread& Kernel::current() {
    if (!tl_current) {
        static Thread interrupt_context;  // HLE calls made from interrupt handlers
        interrupt_context.name = "<interrupt>";
        return interrupt_context;
    }
    return *tl_current;
}

int Kernel::current_id() const { return tl_current ? tl_current->id : 0; }

int Kernel::create_thread(const nx::string& name, std::uint32_t entry, std::uint32_t priority,
                          std::uint32_t stack_size, std::uint32_t attr) {
    auto t = nx::make_unique<Thread>();
    t->id = new_id();
    t->name = name;
    t->entry = entry;
    t->priority = priority;
    t->attr = attr;
    t->stack_size = (std::max<std::uint32_t>(stack_size, 0x200) + 0xFF) & ~0xFFu;
    t->stack = user_heap_.alloc(t->stack_size, 256, true);
    if (!t->stack) return static_cast<int>(kErrorNoMemory);
    memory_.fill(t->stack, 0xFF, t->stack_size);
    const int id = t->id;
    threads_.emplace(id, std::move(t));
    return id;
}

std::uint32_t Kernel::start_thread(int id, std::uint32_t arg_size, std::uint32_t arg_ptr) {
    Thread* t = thread(id);
    if (!t) return kErrorUnknownThread;
    if (t->state != Thread::State::Dormant) return kErrorNotDormant;
    recomp_context_init(&t->ctx);
    // The kernel keeps a 0x100-byte thread info area at the top of the stack and points k0 at it; the SDK's
    // C runtime stores per-thread data there.
    const std::uint32_t k0 = t->stack + t->stack_size - 0x100;
    memory_.fill(k0, 0, 0x100);
    memory_.write32(k0 + 0xC0, static_cast<std::uint32_t>(t->id));
    memory_.write32(k0 + 0xC8, t->stack);
    memory_.write32(k0 + 0xF8, 0xFFFFFFFF);
    memory_.write32(k0 + 0xFC, 0xFFFFFFFF);
    t->ctx.r[26] = k0;
    std::uint32_t sp = k0;
    if (arg_ptr && arg_size) {  // arguments are copied onto the new thread's stack
        sp = (sp - arg_size) & ~0xFu;
        std::memcpy(memory_.ptr(sp), memory_.ptr(arg_ptr), arg_size);
        t->ctx.r[5] = sp;
        sp -= 0x40;
    }
    t->ctx.r[4] = arg_size;
    t->ctx.r[29] = sp;
    t->ctx.r[28] = t->gp;
    t->ctx.r[31] = 0;  // returning from the entry function exits the thread
    make_ready(*t);
    if (t->host.joinable()) t->host.join();
    t->host = nx::thread([this, t] { host_thread_main(t); }, {.name = t->name.c_str()});
    yield_to_higher_priority();
    return 0;
}

void Kernel::host_thread_main(Thread* t) {
    tl_current = t;
    take_baton(t->baton, t->baton_misses);
    std::uint32_t status = 0;
    try {
        const RecompFunc entry = recomp_lookup(t->entry);
        if (!entry) {
            std::fprintf(stderr, "thread %s: no recompiled function at entry 0x%08X\n", t->name.c_str(), t->entry);
            std::abort();
        }
        entry(&t->ctx);
        status = t->ctx.r[2];
    } catch (const ThreadExit&) {
        status = t->exit_status;
    } catch (const GameExit&) {
        quit_ = true;
    }
    t->exit_status = status;
    t->state = Thread::State::Dead;
    if (t->delete_on_exit && t->stack) {  // the thread object stays for WaitThreadEnd; its stack is released
        user_heap_.free(t->stack);
        t->stack = 0;
    }
    for (Thread* w : waiters(WaitType::ThreadEnd, t->id)) wake(*w, status);
    schedule_from(nullptr);  // hand the baton on; this host thread then ends
}

[[noreturn]] void Kernel::exit_current_thread(std::uint32_t status, bool delete_thread) {
    current().exit_status = status;
    current().delete_on_exit |= delete_thread;
    throw ThreadExit{};
}

std::uint32_t Kernel::delete_thread(int id) {
    Thread* t = thread(id);
    if (!t) return kErrorUnknownThread;
    if (t->state != Thread::State::Dead && t->state != Thread::State::Dormant) return kErrorNotDormant;
    if (t->host.joinable()) t->host.join();
    if (t->stack) user_heap_.free(t->stack);
    threads_.erase(id);
    return 0;
}

// ---------------------------------------------------------------------------------------------

void Kernel::make_ready(Thread& t) {
    t.state = Thread::State::Ready;
    t.ready_order = ++ready_counter_;
}

Thread* Kernel::best_ready() const {
    Thread* best = nullptr;
    for (const auto& [_, t] : threads_) {
        if (t->state != Thread::State::Ready) continue;
        if (!best || t->priority < best->priority || (t->priority == best->priority && t->ready_order < best->ready_order))
            best = t.get();
    }
    return best;
}

void Kernel::schedule_from(Thread* from) {
    Thread* next = best_ready();
    if (from && next == from) {
        from->state = Thread::State::Running;
        return;
    }
    if (next) {
        next->state = Thread::State::Running;
        next->baton.release();
    } else {
        idle_baton_.release();
    }
    if (from) {
        take_baton(from->baton, from->baton_misses);  // until the scheduler picks this thread again
        tl_current = from;
    }
}

std::uint32_t Kernel::wait(WaitType type, int object, std::uint32_t timeout_addr, bool callbacks, std::uint64_t delay_us) {
    Thread& t = current();
    if (!tl_current) return 0;  // cannot block in interrupt context
    if (t.kill_requested) throw ThreadExit{};  // terminated while running (no park point reached)
    if (callbacks) check_callbacks();
    t.state = Thread::State::Waiting;
    t.wait = type;
    t.wait_object = object;
    t.wait_callbacks = callbacks;
    t.timeout_addr = timeout_addr;
    t.wake_time = delay_us ? now_us_ + delay_us : timeout_addr ? now_us_ + memory_.read32(timeout_addr) : 0;
    t.wait_result = 0;
    schedule_from(&t);
    if (t.kill_requested) throw ThreadExit{};  // terminated while parked (woken to die)
    if (callbacks) check_callbacks();
    return t.wait_result;
}

void Kernel::wake(Thread& t, std::uint32_t result) {
    if (t.state != Thread::State::Waiting) return;
    if (t.timeout_addr && t.wake_time) {
        const std::uint64_t left = t.wake_time > now_us_ ? t.wake_time - now_us_ : 0;
        memory_.write32(t.timeout_addr, static_cast<std::uint32_t>(left));
    }
    t.wait = WaitType::None;
    t.wait_result = result;
    t.wake_time = 0;
    make_ready(t);
}

void Kernel::yield_to_higher_priority() {
    if (!tl_current) return;
    Thread& cur = *tl_current;
    const Thread* best = best_ready();
    if (!best || best->priority >= cur.priority) return;
    cur.state = Thread::State::Ready;
    cur.ready_order = 0;  // a preempted thread stays first among its priority
    schedule_from(&cur);
}

void Kernel::rotate_ready_queue() {
    if (!tl_current) return;
    make_ready(*tl_current);
    schedule_from(tl_current);
}

nx::vector<Thread*> Kernel::waiters(WaitType type, int object) {
    nx::vector<Thread*> out;
    for (auto& [_, t] : threads_)
        if (t->state == Thread::State::Waiting && t->wait == type && t->wait_object == object) out.push_back(t.get());
    // Wake order: priority, then the order they started waiting.
    std::sort(out.begin(), out.end(), [](const Thread* a, const Thread* b) { return a->priority < b->priority; });
    return out;
}

// ---------------------------------------------------------------------------------------------

int Kernel::create_callback(const nx::string& name, std::uint32_t entry, std::uint32_t arg) {
    auto cb = nx::make_unique<Callback>();
    cb->id = new_id();
    cb->name = name;
    cb->entry = entry;
    cb->arg = arg;
    cb->owner = current_id();
    const int id = cb->id;
    callbacks_.emplace(id, std::move(cb));
    return id;
}

void Kernel::notify_callback(int id, std::uint32_t arg) {
    const auto it = callbacks_.find(id);
    if (it == callbacks_.end()) return;
    ++it->second->notify_count;
    it->second->notify_arg = arg;
    if (Thread* owner = thread(it->second->owner); owner && owner->state == Thread::State::Waiting && owner->wait_callbacks)
        wake(*owner, 0);  // CB waits return to run the callback (approximation of PSP behaviour)
}

void Kernel::check_callbacks() {
    if (!tl_current) return;
    Thread& t = *tl_current;
    // Stepped by id: the guest callback may create or delete callbacks, which invalidates iterators.
    for (auto it = callbacks_.begin(); it != callbacks_.end();) {
        const int id = it->first;
        Callback& cb = *it->second;
        if (cb.owner != t.id || cb.notify_count == 0) {
            ++it;
            continue;
        }
        const Context saved = t.ctx;
        t.ctx.r[4] = static_cast<std::uint32_t>(cb.notify_count);
        t.ctx.r[5] = cb.notify_arg;
        t.ctx.r[6] = cb.arg;
        t.ctx.r[29] -= 0x40;
        cb.notify_count = 0;
        if (const RecompFunc f = recomp_lookup(cb.entry)) f(&t.ctx);
        const bool remove = t.ctx.r[2] != 0;
        t.ctx = saved;
        if (remove) callbacks_.erase(id);
        it = callbacks_.upper_bound(id);
    }
}

void Kernel::dump_threads(std::FILE* out) const {
    static constexpr const char* kStates[] = {"dormant", "ready", "running", "waiting", "dead"};
    static constexpr const char* kWaits[] = {"-", "delay", "sleep", "sema", "eventflag", "threadend", "vblank",
                                             "fpl", "io", "ge", "audio", "other"};
    for (const auto& [id, t] : threads_)
        std::fprintf(out, "  thread %3d %-28s prio %3u %-8s wait %-9s obj %-5d ra %08X\n", id, t->name.c_str(),
                     t->priority, kStates[static_cast<int>(t->state)], kWaits[static_cast<int>(t->wait)],
                     t->wait_object, t->ctx.r[31]);
}

std::uint32_t Kernel::call_guest(std::uint32_t entry, std::initializer_list<std::uint32_t> args) {
    Thread& t = current();
    const Context saved = t.ctx;
    int i = 0;
    for (const std::uint32_t a : args)
        if (i < 8) t.ctx.r[4 + i++] = a;
    t.ctx.r[29] -= 0x40;
    std::uint32_t result = 0;
    if (const RecompFunc f = recomp_lookup(entry)) {
        f(&t.ctx);
        result = t.ctx.r[2];
    } else {
        std::fprintf(stderr, "call_guest: no function at 0x%08X\n", entry);
    }
    t.ctx = saved;
    return result;
}

void Kernel::set_subintr_handler(int intr, int sub, std::uint32_t handler, std::uint32_t arg) {
    auto& s = subintrs_[{intr, sub}];
    s.handler = handler;
    s.arg = arg;
}

void Kernel::enable_subintr(int intr, int sub, bool enable) { subintrs_[{intr, sub}].enabled = enable; }

void Kernel::fire_vblank() {
    ++vblanks_;
    next_vblank_us_ += kVblankPeriodUs;
    for (Thread* w : waiters(WaitType::Vblank, 0)) wake(*w, 0);
    // Collected first: a handler may register another one, which invalidates iterators into subintrs_.
    struct Due {
        int sub;
        std::uint32_t handler, arg;
    };
    nx::small_vector<Due, 8> due;
    for (const auto& [key, s] : subintrs_)
        if (key.first == kVblankInterrupt && s.enabled && s.handler) due.push_back({key.second, s.handler, s.arg});
    for (const Due& d : due) {
        const RecompFunc f = recomp_lookup(d.handler);
        if (!f) continue;
        recomp_context_init(&interrupt_ctx_);
        interrupt_ctx_.r[4] = static_cast<std::uint32_t>(d.sub);
        interrupt_ctx_.r[5] = d.arg;
        interrupt_ctx_.r[29] = interrupt_stack_ + 0x4000 - 0x40;
        f(&interrupt_ctx_);
    }
}

int Kernel::run(std::uint32_t entry, std::uint32_t arg_size, std::uint32_t arg_ptr) {
    nx::timestamp anchor = nx::timestamp::now();  // host time corresponding to virtual time 0 (realtime mode)
    const int main_id = create_thread("module_start", entry, 0x20, 0x40000, 0x80000000);
    thread(main_id)->delete_on_exit = true;  // the loader deletes module_start's thread when it returns
    start_thread(main_id, arg_size, arg_ptr);
    for (;;) {
        schedule_from(nullptr);
        take_baton(idle_baton_, idle_baton_misses_);  // no guest thread is ready
        if (quit_) break;
        const bool alive = std::any_of(threads_.begin(), threads_.end(), [](const auto& kv) {
            return kv.second->state != Thread::State::Dead && kv.second->state != Thread::State::Dormant;
        });
        if (!alive) break;
        // Advance virtual time to the next event.
        std::uint64_t next = next_vblank_us_;
        for (const auto& [_, t] : threads_)
            if (t->state == Thread::State::Waiting && t->wake_time) next = std::min(next, t->wake_time);
        if (realtime && next > now_us_) {
            const nx::timestamp due = anchor + nx::duration::from_micros(static_cast<std::int64_t>(next));
            const nx::timestamp host = nx::timestamp::now();
            behind_us_.store(due < host ? (host - due).micros() : 0, std::memory_order_relaxed);
            if (due > host) nx::this_thread::sleep_for(std::chrono::nanoseconds((due - host).nanos()));
            else if (host - due > nx::duration::from_millis(100)) {  // fell behind
                anchor = host - nx::duration::from_micros(static_cast<std::int64_t>(next));
                ++fell_behind_;
            }
        }
        g_baton_spin_us.store(realtime ? 50 : 300, std::memory_order_relaxed);
        if (!realtime) behind_us_.store(0, std::memory_order_relaxed);
        if (!realtime) {  // unthrottled: when real time resumes, it continues from here
            anchor = nx::timestamp::now() - nx::duration::from_micros(static_cast<std::int64_t>(now_us_));
        }
        now_us_ = std::max(now_us_, next);
        if (now_us_ >= next_vblank_us_) {
            fire_vblank();
            if (on_vblank) on_vblank();
            if (quit_) break;
            if (max_vblanks && vblanks_ >= max_vblanks) break;
        }
        for (auto& [_, t] : threads_) {
            if (t->state != Thread::State::Waiting || !t->wake_time || t->wake_time > now_us_) continue;
            wake(*t, t->wait == WaitType::Delay ? 0 : kErrorWaitTimeout);
        }
    }
    return exit_code_;
}

}  // namespace psprecore::rt
