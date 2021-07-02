#pragma once

#include <exception>
#include <mutex>
#include <chrono>


namespace future {

struct future_ready : public std::exception {};

namespace detail {

// SharedStateBase contains common data and methods for ShartedState<T> and SharedState<void>. 
// These two specializations of SharedState differ in the following ways:
// 1) SharedState<T> has a value_ member of type T, whereas SharedState<void> does not
// 2) The SharedState methods set_value() and get() differ in signature/return type for <T> and <void> 
class SharedStateBase {
public:
    SharedStateBase()
        :   exception_(nullptr),
            ready_(false)
    {}

    ~SharedStateBase() = default;

    // Not copyable or moveable
    SharedStateBase(const SharedStateBase&) = delete;
    SharedStateBase(SharedStateBase&&) = delete;
    SharedStateBase& operator=(const SharedStateBase&) = delete;
    SharedStateBase& operator=(SharedStateBase&&) = delete;

    // Writer methods
    void set_exception(std::exception_ptr eptr) {
        std::scoped_lock<std::mutex> lock(mutex_);

        if (ready_) {
            throw future::future_ready();
        }

        exception_ = std::move(eptr);
        ready_ = true;
        ready_cond_var_.notify_all();
    }

    void set_exception(const std::exception& e) {
        set_exception(std::make_exception_ptr(e));
    }

    // Reader methods
    void wait() {
        std::unique_lock<std::mutex> lock(mutex_);
        ready_cond_var_.wait(lock, [this]{ return ready_; });
    }

    template <typename Rep, typename Period>
    void wait_for(const std::chrono::duration<Rep, Period>& d) {
        std::unique_lock<std::mutex> lock(mutex_);
        ready_cond_var_.wait_for(lock, d, [this]{ return ready_; });
    }

    template <typename Clock, typename Duration>
    void wait_until(const std::chrono::time_point<Clock, Duration>& t) {
        std::unique_lock<std::mutex> lock(mutex_);
        ready_cond_var_.wait_until(lock, t, [this]{ return ready_; });
    }

    bool ready() const {
        std::scoped_lock<std::mutex> lock(mutex_);
        return ready_;
    }

protected:
    std::exception_ptr exception_;
    bool ready_;
    std::condition_variable ready_cond_var_;
    mutable std::mutex mutex_;
};

template <typename T>
class SharedState : public SharedStateBase {

public:

    SharedState() = default;
    ~SharedState() = default;

    // Not copyable or moveable
    SharedState(const SharedState&) = delete;
    SharedState(SharedState&&) = delete;
    SharedState& operator=(const SharedState&) = delete;
    SharedState& operator=(SharedState&&) = delete;

    // Writer method
    void set_value(T value) {
        std::scoped_lock<std::mutex> lock(mutex_);

        if (ready_) {
            throw future::future_ready();
        }

        value_ = std::move(value);
        ready_ = true;
        ready_cond_var_.notify_all();
    }

    // Reader method
    T get() {
        wait();

        // We do not need a lock to access exception_ because after wait() is called
        // the future is ready and the SharedState cannot be written to after the future is
        // ready
        if (exception_) {
            std::rethrow_exception(exception_);
        }
        
        return value_;
    }

private:
    T value_;
};


template <>
class SharedState<void> : public SharedStateBase {

public:

    SharedState() = default;
    ~SharedState() = default;

    // Not copyable or moveable
    SharedState(const SharedState&) = delete;
    SharedState(SharedState&&) = delete;
    SharedState& operator=(const SharedState&) = delete;
    SharedState& operator=(SharedState&&) = delete;

    // Writer method
    void set_value() {
        std::scoped_lock<std::mutex> lock(mutex_);

        if (ready_) {
            throw future::future_ready();
        }

        ready_ = true;
        ready_cond_var_.notify_all();
    }

    // Reader method
    void get() {
        wait();

        // We do not need a lock to access exception_ because after wait() is called
        // the future is ready and the SharedState cannot be written to after the future is
        // ready
        if (exception_) {
            std::rethrow_exception(exception_);
        }
    }
};

} // namespace detail

} // namespace future