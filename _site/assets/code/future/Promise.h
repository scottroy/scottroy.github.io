#pragma once

#include "SharedState.h"
#include "Future.h"
#include <memory>
#include <string>

namespace future {

struct broken_promise : public std::exception {
private:
    const char * msg_;
public:
    broken_promise(const char *msg = "")
        : msg_(msg) {}

    virtual char const* what() const noexcept override { return msg_; }
};

namespace detail {
template <typename T>
class PromiseBase {
public:

    PromiseBase()
        : state_ptr_(std::make_shared<detail::SharedState<T>>())
    {}

    ~PromiseBase() {
        abandon("Promise was destroyed.");
    }
    
    // Moveable
    PromiseBase(PromiseBase&& rhs) noexcept
        : state_ptr_(std::exchange(rhs.state_ptr_, nullptr))
    {}

    PromiseBase& operator=(PromiseBase&& rhs) noexcept {
        if (this != &rhs) {
            abandon("Promise was moved.");
            state_ptr_ = std::exchange(rhs.state_ptr_, nullptr);
        }

        return *this;
    }

    // Not copyable
    PromiseBase(const PromiseBase&) = delete;
    PromiseBase& operator=(const PromiseBase&) = delete;

    void set_exception(const std::exception& e) {
        throw_if_broken_();
        state_ptr_->set_exception(e);
    }

    void set_exception(std::exception_ptr eptr) {
        throw_if_broken_();
        state_ptr_->set_exception(eptr);
    }

    void abandon(const char* msg = "") noexcept {

        if (state_ptr_ == nullptr) {
            return;
        }

        try {
            state_ptr_->set_exception(std::make_exception_ptr(broken_promise(msg)));
        } catch (...) {
            // Debug only: future must have been ready before abandon attempts to set
            assert(state_ptr_->ready());
        }

        // Sever promise from SharedState (decrements its reference count)
        state_ptr_ = nullptr;
    }

    Future<T> get_future() const {
        throw_if_broken_();
        return Future<T>(state_ptr_);
    }

protected:
    std::shared_ptr<detail::SharedState<T>> state_ptr_;

    inline void throw_if_broken_() const {
        if (state_ptr_ == nullptr) {
            throw broken_promise();
        }
    }
};
} // namespace detail

template <typename T>
class Promise : public detail::PromiseBase<T> {
public:

    Promise() = default;
    
    // Moveable
    Promise(Promise&&) = default;
    Promise& operator=(Promise&&) = default;

    // Not copyable
    Promise(const Promise&) = delete;
    Promise& operator=(const Promise&) = delete;

    ~Promise() = default;

    void set_value(T value) {
        this->throw_if_broken_();
        this->state_ptr_->set_value(std::move(value));
    }
};


template <>
class Promise<void> : public detail::PromiseBase<void> {
public:

    Promise() = default;
    
    // Moveable
    Promise(Promise&&) = default;
    Promise& operator=(Promise&&) = default;

    // Not copyable
    Promise(const Promise&) = delete;
    Promise& operator=(const Promise&) = delete;

    ~Promise() = default;

    void set_value() {
        this->throw_if_broken_();
        this->state_ptr_->set_value();
    }

};

} // namespace future

