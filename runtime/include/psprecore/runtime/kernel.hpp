#pragma once

// PSP kernel emulation (HLE): threads, scheduling, virtual time and synchronisation objects.
//
// Every guest thread runs on its own host thread, but only one at a time: a thread runs while it holds its
// "baton" and hands it on whenever it blocks or is preempted. This gives the PSP's single-core priority
// semantics deterministically. The host's main thread is the idle loop: when no guest thread is ready it
// advances virtual time to the next event (timeouts, vblank), fires it, and hands the baton on again.

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <memory>

#include "psprecore/recomp/recomp.h"
#include "psprecore/runtime/memory.hpp"
#include "core/foundation/core/callable.h"
#include "core/foundation/core/foundation.h"
#include "core/foundation/containers/map.h"
#include "core/foundation/strings/utf8_string.h"
#include "core/foundation/threading/sync.h"
#include "core/foundation/threading/thread.h"

namespace psprecore::rt {

// PSP kernel error codes used by the HLE functions.
enum : std::uint32_t {
    kErrorIllegalPriority = 0x80020193,
    kErrorUnknownThread = 0x80020198,
    kErrorUnknownSema = 0x80020199,
    kErrorUnknownEventFlag = 0x8002019A,
    kErrorUnknownFpl = 0x8002019B,
    kErrorUnknownCallback = 0x800201A1,
    kErrorWaitTimeout = 0x800201A8,
    kErrorWaitCancel = 0x800201A9,
    kErrorWaitDelete = 0x800201B5,
    kErrorNotDormant = 0x800201A4,
    kErrorDormant = 0x800201A2,
    kErrorSemaOverflow = 0x800201BD,
    kErrorSemaZero = 0x800201AD,
    kErrorEventFlagPollFailed = 0x800201AF,
    kErrorNoMemory = 0x80020190,
};

enum class WaitType : std::uint8_t { None, Delay, Sleep, Sema, EventFlag, ThreadEnd, Vblank, Fpl, Io, Ge, Audio, Other };

struct Thread {
    int id = 0;
    nx::string name;
    std::uint32_t entry = 0, priority = 0, attr = 0, stack = 0, stack_size = 0, gp = 0;
    Context ctx{};
    enum class State : std::uint8_t { Dormant, Ready, Running, Waiting, Dead } state = State::Dormant;

    WaitType wait = WaitType::None;
    int wait_object = 0;
    std::uint64_t wake_time = 0;  // virtual µs; 0 = no timeout
    std::uint32_t timeout_addr = 0;
    bool wait_callbacks = false;
    std::uint32_t wait_result = 0;
    std::uint32_t wait_value = 0, wait_mode = 0, wait_out = 0;  // object-specific (event flag bits, sema count…)

    int wakeup_requests = 0;
    std::uint32_t exit_status = 0;
    bool delete_on_exit = false;  // ExitDeleteThread / loader-owned module_start thread
    bool kill_requested = false;  // TerminateDeleteThread: unwind the host thread at the next wait point
    std::uint64_t ready_order = 0;  // FIFO position among equal priorities

    nx::semaphore baton{0};  // released when the scheduler hands this thread the guest CPU
    std::uint8_t baton_misses = 0;  // recent waits for the baton that outlasted the spin (kernel.cpp take_baton)
    nx::thread host;
};

struct Semaphore {
    int id = 0;
    nx::string name;
    std::uint32_t attr = 0;
    int count = 0, max = 0;
};

struct EventFlag {
    int id = 0;
    nx::string name;
    std::uint32_t attr = 0, bits = 0;
};

struct Fpl {
    int id = 0;
    nx::string name;
    std::uint32_t attr = 0, block_size = 0, base = 0;
    nx::vector<bool> used;
};

struct Callback {
    int id = 0;
    nx::string name;
    std::uint32_t entry = 0, arg = 0;
    int owner = 0;
    int notify_count = 0;
    std::uint32_t notify_arg = 0;
};

// Thrown through recompiled frames to unwind a guest thread that exits from inside guest code.
struct ThreadExit {};
// Thrown when the game calls sceKernelExitGame.
struct GameExit {};

class Kernel {
public:
    explicit Kernel(GuestMemory& memory);
    ~Kernel();

    GuestMemory& memory() { return memory_; }
    GuestHeap& user_heap() { return user_heap_; }
    // Guest memory outside the game's partition, for data a port adds without moving the game's own allocations.
    GuestHeap& kernel_heap() { return kernel_heap_; }

    // ---- threads ----
    int create_thread(const nx::string& name, std::uint32_t entry, std::uint32_t priority, std::uint32_t stack_size,
                      std::uint32_t attr);
    std::uint32_t start_thread(int id, std::uint32_t arg_size, std::uint32_t arg_ptr);
    [[noreturn]] void exit_current_thread(std::uint32_t status, bool delete_thread);
    std::uint32_t delete_thread(int id);
    Thread* thread(int id);
    Thread& current();
    int current_id() const;
    void dump_threads(std::FILE* out) const;  // state, wait reason and return address of every thread

    // ---- blocking and waking (called from HLE functions on the current guest thread) ----
    // Blocks the current thread; returns the wait result. `timeout_addr` points to a guest u32 timeout in µs
    // (updated with the time left on wake); `delay_us` is a plain timeout (sceKernelDelayThread).
    std::uint32_t wait(WaitType type, int object, std::uint32_t timeout_addr, bool callbacks, std::uint64_t delay_us = 0);
    void wake(Thread& t, std::uint32_t result);
    void yield_to_higher_priority();  // preempt the current thread if a better one became ready
    void schedule_from(Thread* from);  // hand the baton to the best ready thread (or the idle loop)
    void rotate_ready_queue();        // yield to equal-priority threads
    nx::vector<Thread*> waiters(WaitType type, int object);

    // ---- objects ----
    template <typename T>
    nx::map<int, nx::unique_ptr<T>>& objects();
    int new_id() { return next_id_++; }
    void check_callbacks();  // run pending callbacks for the current thread (CB waits)
    // Calls guest function `entry` on the current thread (EABI args a0-a3, t0-t3) and returns v0. The caller's
    // registers are preserved, so HLE functions can use it for guest callbacks (e.g. ring buffer fills).
    std::uint32_t call_guest(std::uint32_t entry, std::initializer_list<std::uint32_t> args);
    int create_callback(const nx::string& name, std::uint32_t entry, std::uint32_t arg);
    void notify_callback(int id, std::uint32_t arg);

    // ---- time and interrupts ----
    std::uint64_t now() const { return now_us_; }
    void advance(std::uint64_t us) { now_us_ += us; }
    std::uint64_t vblank_count() const { return vblanks_; }
    void set_subintr_handler(int intr, int sub, std::uint32_t handler, std::uint32_t arg);
    void enable_subintr(int intr, int sub, bool enable);

    // ---- main loop ----
    // Starts the module entry thread and runs until every thread has exited or the game quits.
    int run(std::uint32_t entry, std::uint32_t arg_size, std::uint32_t arg_ptr);

    bool trace_hle = false;
    bool realtime = false;           // pace virtual time to the host clock (windowed play)
    nx::unique_function<void()> on_vblank;  // runs on the host main thread at every vblank (present, input)
    void request_quit() { quit_ = true; }
    bool quit_requested() const { return quit_; }
    std::uint64_t max_vblanks = 0;  // stop after this many vblanks (0 = run forever)
    // Realtime mode: how often the guest fell more than 100 ms behind the host clock (time was dropped).
    std::uint64_t fell_behind_count() const { return fell_behind_; }
    // In real time: how far virtual time lags behind the host clock (microseconds; 0 when on time or ahead).
    std::int64_t behind_us() const { return behind_us_.load(std::memory_order_relaxed); }

private:
    Thread* best_ready() const;
    void make_ready(Thread& t);
    void host_thread_main(Thread* t);
    void fire_vblank();

    std::uint64_t fell_behind_ = 0;
    std::atomic<std::int64_t> behind_us_{0};
    GuestMemory& memory_;
    GuestHeap user_heap_;
    GuestHeap kernel_heap_;  // kernel-owned allocations (interrupt stack), outside the user partition
    nx::map<int, nx::unique_ptr<Thread>> threads_;
    nx::map<int, nx::unique_ptr<Semaphore>> semas_;
    nx::map<int, nx::unique_ptr<EventFlag>> event_flags_;
    nx::map<int, nx::unique_ptr<Fpl>> fpls_;
    nx::map<int, nx::unique_ptr<Callback>> callbacks_;
    struct SubIntr {
        std::uint32_t handler = 0, arg = 0;
        bool enabled = false;
    };
    nx::map<std::pair<int, int>, SubIntr> subintrs_;
    int next_id_ = 0x100;
    std::uint64_t ready_counter_ = 0;
    std::uint64_t now_us_ = 0, next_vblank_us_ = 16683, vblanks_ = 0;
    nx::semaphore idle_baton_{0};
    std::uint8_t idle_baton_misses_ = 0;
    Context interrupt_ctx_{};
    std::uint32_t interrupt_stack_ = 0;
    bool quit_ = false;
    int exit_code_ = 0;
};

template <> inline nx::map<int, nx::unique_ptr<Semaphore>>& Kernel::objects() { return semas_; }
template <> inline nx::map<int, nx::unique_ptr<EventFlag>>& Kernel::objects() { return event_flags_; }
template <> inline nx::map<int, nx::unique_ptr<Fpl>>& Kernel::objects() { return fpls_; }
template <> inline nx::map<int, nx::unique_ptr<Callback>>& Kernel::objects() { return callbacks_; }

Kernel& kernel();  // the single kernel instance (set by the host before running)
void set_kernel(Kernel* k);

}  // namespace psprecore::rt
