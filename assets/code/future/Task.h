#pragma once
#include "Promise.h"
#include <thread>
#include <type_traits>
#include <iostream>

namespace future {

template <typename R>
class Task {
public:

    template <typename Callable>
    Task(Callable task)
        : task_(std::move(task))
    {}

    Task(Task&&) = default;
    Task& operator=(Task&&) = default;

    Task(const Task&) = delete;
    Task& operator=(const Task&) = delete;

    ~Task() {
        if (thread_.joinable()) {
            thread_.join();
        }
    }

    void start() {
        try {
            // Store value separately so we don't lock the SharedState while its computed
            auto value = task_(); 
            promise_.set_value(value);
        } catch(...) {
            promise_.set_exception(std::current_exception());
        };
    }

    void start_on_new_thread() {
        thread_ = std::thread([this]{ start(); });
    }

    inline Future<R> get_future() const {
        return promise_.get_future();
    }

    template <typename Callable>
    auto then(Callable next) {
        using R2 = typename std::result_of_t<Callable(Future<R>)>;

        auto task_ptr = std::make_shared<Task<R>>(std::move(*this));

        return Task<R2>([task_ptr=task_ptr, next=std::move(next)] { 
            task_ptr->start_on_new_thread();
            auto res = task_ptr->get_future();
            res.wait();
            return next(std::move(res));
        });
    }

private:
    Promise<R> promise_;
    std::function<R()> task_;
    std::thread thread_;
};

// void specialization
// Only difference is in start() method
template <>
class Task<void> {
public:

    template <typename Callable>
    Task(Callable task)
        : task_(std::move(task))
    {}

    Task(Task&&) = default;
    Task& operator=(Task&&) = default;

    Task(const Task&) = delete;
    Task& operator=(const Task&) = delete;

    ~Task() {
        if (thread_.joinable()) {
            thread_.join();
        }
    }

    void start() {
        try {
            // Store value separately so we don't lock the SharedState while its computed
            task_(); 
            promise_.set_value();
        } catch(...) {
            promise_.set_exception(std::current_exception());
        };
    }

    void start_on_new_thread() {
        thread_ = std::thread([this]{ start(); });
    }

    inline Future<void> get_future() const {
        return promise_.get_future();
    }

    template <typename Callable>
    auto then(Callable next) {
        using R2 = typename std::result_of_t<Callable(Future<void>)>;

        auto task_ptr = std::make_shared<Task<void>>(std::move(*this));

        return Task<R2>([task_ptr=task_ptr, next=std::move(next)] { 
            task_ptr->start_on_new_thread();
            auto res = task_ptr->get_future();
            res.wait();
            return next(std::move(res));
        });
    }

private:
    Promise<void> promise_;
    std::function<void()> task_;
    std::thread thread_;
};

template <typename Callable>
auto MakeTask(Callable task) {
    return Task<typename std::result_of_t<Callable()>>(std::move(task));
};


// Task parametrized by Callable instead of return type
// Removes std::function and then method does not create a ptr
template <typename Callable>
class Task2 {

public:
    using R = typename std::result_of<Callable()>::type;

    Task2(Callable task)
        : task_(std::move(task))
    {}

    Task2(Task2&&) = default;
    Task2& operator=(Task2&&) = default;

    Task2(const Task2&) = delete;
    Task2& operator=(const Task2&) = delete;

    ~Task2() {
        if (thread_.joinable()) {
            thread_.join();
        }
    }

    void start() {
        try {
            // Store value separately so we don't lock the SharedState while its computed
            auto value = task_(); 
            promise_.set_value(value);
        } catch(...) {
            promise_.set_exception(std::current_exception());
        };
    }

    void start_on_new_thread() {
        thread_ = std::thread([this]{ start(); });
    }

    inline Future<R> get_future() const {
        return promise_.get_future();
    }

    template <typename Callable2>
    auto then(Callable2 next) {

        auto lam = [task=std::move(*this), next=std::move(next)]() mutable { 
            task.start_on_new_thread();
            auto res = task.get_future();
            res.wait();
            return next(std::move(res));
        };

        return Task2<decltype(lam)>(std::move(lam));
    }

private:
    Promise<R> promise_;
    Callable task_;
    std::thread thread_;
};

template <typename Callable>
auto MakeTask2(Callable task) {
    return Task2<decltype(task)>(std::move(task));
};

} // namespace future

