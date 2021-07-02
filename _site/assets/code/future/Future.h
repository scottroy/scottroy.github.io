#pragma once
#include <exception>
#include <memory>
#include "SharedState.h"

namespace future {

struct invalid_future : public std::exception {};

// FutureBase contains common data and methods for
// Future<T>, Future<void>, SharedFuture<T>, and SharedFuture<void>
// Future and SharedFuture differ on the definition of copy operations.
// The <T> and <void> versions differ on the specification of the get() method.
namespace detail {

template <typename T>
class FutureBase {
public:
    FutureBase()
        : state_ptr_(nullptr)
    {}

    FutureBase(std::shared_ptr<detail::SharedState<T>> state_ptr)
        : state_ptr_(state_ptr)
    {}

    FutureBase(FutureBase&& rhs) = default;
    FutureBase& operator=(FutureBase&& rhs) = default;

    FutureBase(const FutureBase& rhs) = default;
    FutureBase& operator=(const FutureBase& rhs) = default;

    ~FutureBase() = default;

    inline bool valid() const noexcept {
        return state_ptr_ != nullptr;
    }

    inline bool ready() const {
        throw_if_not_valid_();
        return state_ptr_->ready();
    }

    inline void wait() const {
        throw_if_not_valid_();
        state_ptr_->wait();
    }

    template <typename Rep, typename Period>
    inline void wait_for(const std::chrono::duration<Rep, Period>& d) const {
        throw_if_not_valid_();
        state_ptr_->wait_for(d);
    }

    template <typename Clock, typename Duration>
    inline void wait_until(const std::chrono::time_point<Clock, Duration>& t) const {
        throw_if_not_valid_();
        state_ptr_->wait_until(t);
    }

    inline void sever() {
        state_ptr_ = nullptr;
    }

protected:
    std::shared_ptr<detail::SharedState<T>> state_ptr_;
    inline void throw_if_not_valid_() const {
        if (state_ptr_ == nullptr) {
            throw invalid_future();
        }
    }
};

} // namesapce detail

// We define SharedFuture before Future because future
// has a make_shared method that returns a SharedFuture.
// A forward declaration is not sufficient because a pointer is not being returned.
// The size of the object must be known.

template <typename T>
class SharedFuture : public detail::FutureBase<T> {

public:

    SharedFuture()
        : detail::FutureBase<T>()
    {}

    SharedFuture(std::shared_ptr<detail::SharedState<T>> state_ptr)
        : detail::FutureBase<T>(state_ptr)
    {}

    SharedFuture(SharedFuture&& rhs) = default;
    SharedFuture& operator=(SharedFuture&& rhs) = default;

    SharedFuture(const SharedFuture& rhs) = default;
    SharedFuture& operator=(const SharedFuture& rhs) = default;

    ~SharedFuture() = default;

    inline T get() const {
        this->throw_if_not_valid_();
        return this->state_ptr_->get();
    }
};

template<>
class SharedFuture<void> : public detail::FutureBase<void> {

public:

    SharedFuture()
        : detail::FutureBase<void>()
    {}

    SharedFuture(std::shared_ptr<detail::SharedState<void>> state_ptr)
        : detail::FutureBase<void>(state_ptr)
    {}

    SharedFuture(SharedFuture&& rhs) = default;
    SharedFuture& operator=(SharedFuture&& rhs) = default;

    SharedFuture(const SharedFuture& rhs) = default;
    SharedFuture& operator=(const SharedFuture& rhs) = default;

    ~SharedFuture() = default;

    inline void get() const {
        this->throw_if_not_valid_();
        this->state_ptr_->get();
    }
};

template <typename T>
class Future : public detail::FutureBase<T> {

public:

    Future()
        : detail::FutureBase<T>()
    {}

    Future(std::shared_ptr<detail::SharedState<T>> state_ptr)
        : detail::FutureBase<T>(state_ptr)
    {}

    Future(Future&& rhs) = default;
    Future& operator=(Future&& rhs) = default;

    Future(const Future& rhs) = delete;
    Future& operator=(const Future& rhs) = delete;

    ~Future() = default;

    inline T get() const {
        this->throw_if_not_valid_();
        return this->state_ptr_->get();
    }

    SharedFuture<T> make_shared() const {
        return SharedFuture<T>(this->state_ptr_);
    }
};

template <>
class Future<void> : public detail::FutureBase<void> {

public:

    Future()
        : detail::FutureBase<void>()
    {}

    Future(std::shared_ptr<detail::SharedState<void>> state_ptr)
        : detail::FutureBase<void>(state_ptr)
    {}

    Future(Future&& rhs) = default;
    Future& operator=(Future&& rhs) = default;

    Future(const Future& rhs) = delete;
    Future& operator=(const Future& rhs) = delete;

    ~Future() = default;

    inline void get() const {
        this->throw_if_not_valid_();
        this->state_ptr_->get();
    }

    SharedFuture<void> make_shared() const {
        return SharedFuture<void>(this->state_ptr_);
    }
};



} // namespace future