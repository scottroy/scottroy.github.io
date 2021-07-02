#include "gtest/gtest.h"
#include "SharedState.h"
#include "Promise.h"
#include "Future.h"
#include <string>


using namespace future;

TEST(FutureTest, SharedFutureCopyNonvoidType) {
    Promise<std::string> promise;
    auto future = promise.get_future().make_shared();

    SharedFuture<std::string> future2;
    SharedFuture<std::string> future3;

    promise.set_value("from the other world");

    EXPECT_TRUE(future.valid());
    EXPECT_FALSE(future2.valid());
    EXPECT_FALSE(future3.valid());
    future2 = future;
    future3 = future2;
    EXPECT_TRUE(future.valid());
    EXPECT_TRUE(future2.valid());
    EXPECT_TRUE(future3.valid());
    
    EXPECT_EQ(future.get(), "from the other world");
    EXPECT_EQ(future2.get(), "from the other world");
    EXPECT_EQ(future3.get(), "from the other world");

    future.sever();
    EXPECT_EQ(future2.get(), "from the other world");
    EXPECT_EQ(future3.get(), "from the other world");
}

TEST(FutureTest, SharedFutureCopyVoidType) {
    Promise<void> promise;
    auto future = promise.get_future().make_shared();

    SharedFuture<void> future2;
    SharedFuture<void> future3;

    promise.set_exception(std::exception());

    EXPECT_TRUE(future.valid());
    EXPECT_FALSE(future2.valid());
    EXPECT_FALSE(future3.valid());
    future2 = future;
    future3 = future2;
    EXPECT_TRUE(future.valid());
    EXPECT_TRUE(future2.valid());
    EXPECT_TRUE(future3.valid());
    
    EXPECT_THROW(future.get(), std::exception);
    EXPECT_THROW(future2.get(), std::exception);
    EXPECT_THROW(future3.get(), std::exception);

    future.sever();
    EXPECT_THROW(future2.get(), std::exception);
    EXPECT_THROW(future3.get(), std::exception);
}


TEST(FutureTest, FutureMoveCtorNonvoidType) {
    Promise<std::string> promise;
    auto future = promise.get_future();
    promise.set_value("from the other world");

    EXPECT_TRUE(future.valid());
    Future<std::string> alternate(std::move(future));
    EXPECT_FALSE(future.valid());
    EXPECT_TRUE(alternate.valid());
    EXPECT_EQ(alternate.get(), "from the other world");
}

TEST(FutureTest, FutureMoveCtorVoidType) {
    Promise<void> promise;
    auto future = promise.get_future();
    promise.set_exception(std::exception());

    EXPECT_TRUE(future.valid());
    Future<void> alternate(std::move(future));
    EXPECT_FALSE(future.valid());
    EXPECT_TRUE(alternate.valid());
    EXPECT_THROW(alternate.get(), std::exception);
}

TEST(FutureTest, FutureMoveNonvoidType) {
    Promise<std::string> promise;
    auto future = promise.get_future();
    Future<std::string> alternate;

    EXPECT_TRUE(future.valid());
    EXPECT_FALSE(alternate.valid());
    alternate = std::move(future);
    EXPECT_TRUE(alternate.valid());
    EXPECT_FALSE(future.valid());
    
    promise.set_value("from the other world");
    EXPECT_EQ(alternate.get(), "from the other world");
}

TEST(FutureTest, FutureMoveVoidType) {
    Promise<void> promise;
    auto future = promise.get_future();
    Future<void> alternate;
    
    EXPECT_TRUE(future.valid());
    EXPECT_FALSE(alternate.valid());
    alternate = std::move(future);
    EXPECT_TRUE(alternate.valid());
    EXPECT_FALSE(future.valid());
    
    promise.set_exception(std::exception());
    EXPECT_THROW(alternate.get(), std::exception);
}

TEST(FutureTest, PromiseMoveNonvoidType) {
    Promise<std::string> promise;
    auto future = promise.get_future();

    Promise<std::string> another;
    another.set_value("another");

    promise = std::move(another);

    EXPECT_THROW(future.get(), broken_promise);
    EXPECT_THROW(another.set_value(""), broken_promise);

    EXPECT_EQ(promise.get_future().get(), "another");
}

TEST(FutureTest, PromiseMoveVoidType) {
    Promise<void> promise;
    auto future = promise.get_future();

    Promise<void> another;
    another.set_exception(std::out_of_range("exception"));

    promise = std::move(another);

    EXPECT_THROW(future.get(), broken_promise);
    EXPECT_THROW(another.set_value(), broken_promise);
    EXPECT_THROW(promise.get_future().get(), std::exception);
}

TEST(FutureTest, PromiseMoveCtorNonvoidType) {
    Promise<std::string> promise;
    auto future = promise.get_future();

    auto moved_promise(std::move(promise));
    EXPECT_THROW(promise.set_value("touching carcass"), broken_promise);

    moved_promise.set_value("follow me");
    EXPECT_EQ(future.get(), "follow me");
}

TEST(FutureTest, PromiseMoveCtorVoidType) {
    Promise<void> promise;
    auto future = promise.get_future();

    auto moved_promise(std::move(promise));
    EXPECT_THROW(promise.set_value(), broken_promise);

    moved_promise.set_exception(std::make_exception_ptr(std::out_of_range("")));
    EXPECT_THROW(future.get(), std::out_of_range);
}

TEST(FutureTest, SeverFutureNonvoidType) {

    Promise<std::string> promise;
    auto future = promise.get_future();
    auto shared_future = promise.get_future().make_shared();

    EXPECT_TRUE(future.valid());
    EXPECT_TRUE(shared_future.valid());
    future.sever();
    EXPECT_FALSE(future.valid());
    EXPECT_TRUE(promise.get_future().valid());
    EXPECT_TRUE(shared_future.valid());

    EXPECT_NO_THROW(future.sever());
    EXPECT_THROW(future.ready(), invalid_future);
    EXPECT_THROW(future.get(), invalid_future);
    EXPECT_THROW(future.wait(), invalid_future);
    EXPECT_THROW(future.wait_for(std::chrono::seconds(1)), invalid_future);
    EXPECT_THROW(future.wait_until(std::chrono::steady_clock::now()), invalid_future);

    shared_future.sever();
    EXPECT_FALSE(shared_future.valid());
    EXPECT_NO_THROW(shared_future.sever());
    EXPECT_THROW(shared_future.get(), invalid_future);
    EXPECT_THROW(shared_future.ready(), invalid_future);
    EXPECT_THROW(shared_future.wait(), invalid_future);
    EXPECT_THROW(shared_future.wait_for(std::chrono::seconds(1)), invalid_future);
    EXPECT_THROW(shared_future.wait_until(std::chrono::steady_clock::now()), invalid_future);
}

TEST(FutureTest, SeverFutureVoidType) {
    
    Promise<void> promise;
    auto future = promise.get_future();
    auto shared_future = promise.get_future().make_shared();

    EXPECT_TRUE(future.valid());
    EXPECT_TRUE(shared_future.valid());
    future.sever();
    EXPECT_FALSE(future.valid());
    EXPECT_TRUE(promise.get_future().valid());
    EXPECT_TRUE(shared_future.valid());

    EXPECT_NO_THROW(future.sever());
    EXPECT_THROW(future.ready(), invalid_future);
    EXPECT_THROW(future.get(), invalid_future);
    EXPECT_THROW(future.wait(), invalid_future);
    EXPECT_THROW(future.wait_for(std::chrono::seconds(1)), invalid_future);
    EXPECT_THROW(future.wait_until(std::chrono::steady_clock::now()), invalid_future);

    shared_future.sever();
    EXPECT_FALSE(shared_future.valid());
    EXPECT_NO_THROW(shared_future.sever());
    EXPECT_THROW(shared_future.get(), invalid_future);
    EXPECT_THROW(shared_future.ready(), invalid_future);
    EXPECT_THROW(shared_future.wait(), invalid_future);
    EXPECT_THROW(shared_future.wait_for(std::chrono::seconds(1)), invalid_future);
    EXPECT_THROW(shared_future.wait_until(std::chrono::steady_clock::now()), invalid_future);
}


TEST(FutureTest, AbandonPromiseNonvoidType) {

    Promise<std::string> promise;
    auto future = promise.get_future();
    EXPECT_FALSE(future.ready());

    promise.abandon();
    EXPECT_NO_THROW(promise.abandon());
    EXPECT_THROW(promise.get_future(), broken_promise);
    EXPECT_THROW(promise.set_value("a promise"), broken_promise);
    EXPECT_THROW(promise.set_exception(std::out_of_range("oh no!")), broken_promise);
    EXPECT_THROW(promise.set_exception(std::make_exception_ptr(std::out_of_range("oh no!"))), broken_promise);
    
    EXPECT_TRUE(future.ready());
    EXPECT_THROW(future.get(), broken_promise);
    EXPECT_THROW(future.make_shared().get(), broken_promise);
}

TEST(FutureTest, AbandonPromiseVoidType) {

    Promise<void> promise;
    auto future = promise.get_future();
    EXPECT_FALSE(future.ready());

    promise.abandon();
    EXPECT_NO_THROW(promise.abandon());
    EXPECT_THROW(promise.get_future(), broken_promise);
    EXPECT_THROW(promise.set_value(), broken_promise);
    EXPECT_THROW(promise.set_exception(std::out_of_range("oh no!")), broken_promise);
    EXPECT_THROW(promise.set_exception(std::make_exception_ptr(std::out_of_range("oh no!"))), broken_promise);
    
    EXPECT_TRUE(future.ready());
    EXPECT_THROW(future.get(), broken_promise);
    EXPECT_THROW(future.make_shared().get(), broken_promise);
}

TEST(FutureTest, SetAndGetValueNonvoidType) {

    Promise<std::string> promise;

    auto future = promise.get_future();
    auto shared_future = promise.get_future().make_shared();

    EXPECT_FALSE(future.ready());
    EXPECT_FALSE(shared_future.ready());

    promise.set_value("be true");

    EXPECT_TRUE(future.ready());
    EXPECT_TRUE(shared_future.ready());
    
    ASSERT_EQ(future.get(), "be true");
    ASSERT_EQ(shared_future.get(), "be true");

    EXPECT_THROW(promise.set_value("be false"), future_ready);
    EXPECT_THROW(promise.set_exception(std::out_of_range("ruh roh again")), future_ready);
    EXPECT_THROW(promise.set_exception(std::make_exception_ptr(std::out_of_range("ruh roh again"))), future_ready);
}

TEST(FutureTest, SetAndGetValueVoidType) {

    Promise<void> promise;

    auto future = promise.get_future();
    auto shared_future = promise.get_future().make_shared();

    EXPECT_FALSE(promise.get_future().ready());
    EXPECT_FALSE(shared_future.ready());

    promise.set_value();

    EXPECT_TRUE(promise.get_future().ready());
    EXPECT_TRUE(shared_future.ready());
    
    EXPECT_NO_THROW(promise.get_future().get());
    EXPECT_NO_THROW(shared_future.get());

    EXPECT_THROW(promise.set_value(), future_ready);
    EXPECT_THROW(promise.set_exception(std::out_of_range("ruh roh again")), future_ready);
    EXPECT_THROW(promise.set_exception(std::make_exception_ptr(std::out_of_range("ruh roh again"))), future_ready);
}

TEST(FutureTest, SetAndGetExceptionNonvoidType) {

    Promise<std::string> promise;

    auto future = promise.get_future();
    auto shared_future = promise.get_future().make_shared();

    EXPECT_FALSE(future.ready());
    EXPECT_FALSE(shared_future.ready());

    promise.set_exception(std::out_of_range("ruh roh"));

    EXPECT_TRUE(future.ready());
    EXPECT_TRUE(shared_future.ready());

    EXPECT_THROW(future.get(), std::exception);
    EXPECT_THROW(shared_future.get(), std::exception);
    
    EXPECT_THROW(promise.set_value("be false"), future_ready);
    EXPECT_THROW(promise.set_exception(std::out_of_range("ruh roh again")), future_ready);
    EXPECT_THROW(promise.set_exception(std::make_exception_ptr(std::out_of_range("ruh roh again"))), future_ready);
}

TEST(FutureTest, SetAndGetExceptionVoidType) {

    Promise<void> promise;

    auto future = promise.get_future();
    auto shared_future = promise.get_future().make_shared();

    EXPECT_FALSE(future.ready());
    EXPECT_FALSE(shared_future.ready());

    promise.set_exception(std::out_of_range("ruh roh"));

    EXPECT_TRUE(future.ready());
    EXPECT_TRUE(shared_future.ready());

    EXPECT_THROW(future.get(), std::exception);
    EXPECT_THROW(shared_future.get(), std::exception);
    
    EXPECT_THROW(promise.set_value(), future_ready);
    EXPECT_THROW(promise.set_exception(std::out_of_range("ruh roh again")), future_ready);
    EXPECT_THROW(promise.set_exception(std::make_exception_ptr(std::out_of_range("ruh roh again"))), future_ready);
}

TEST(FutureTest, SetAndGetExceptionPtrNonvoidType) {

    Promise<std::string> promise;

    auto future = promise.get_future();
    auto shared_future = promise.get_future().make_shared();

    EXPECT_FALSE(future.ready());
    EXPECT_FALSE(shared_future.ready());

    promise.set_exception(std::make_exception_ptr(std::out_of_range("ruh roh")));

    EXPECT_TRUE(future.ready());
    EXPECT_TRUE(shared_future.ready());

    EXPECT_THROW(future.get(), std::out_of_range);
    EXPECT_THROW(shared_future.get(), std::out_of_range);
    
    EXPECT_THROW(promise.set_value("be false"), future_ready);
    EXPECT_THROW(promise.set_exception(std::out_of_range("ruh roh again")), future_ready);
    EXPECT_THROW(promise.set_exception(std::make_exception_ptr(std::out_of_range("ruh roh again"))), future_ready);
}

TEST(FutureTest, SetAndGetExceptionPtrVoidType) {

    Promise<void> promise;

    auto future = promise.get_future();
    auto shared_future = promise.get_future().make_shared();

    EXPECT_FALSE(future.ready());
    EXPECT_FALSE(shared_future.ready());

    promise.set_exception(std::make_exception_ptr(std::out_of_range("ruh roh")));

    EXPECT_TRUE(future.ready());
    EXPECT_TRUE(shared_future.ready());

    EXPECT_THROW(future.get(), std::out_of_range);
    EXPECT_THROW(shared_future.get(), std::out_of_range);
    
    EXPECT_THROW(promise.set_value(), future_ready);
    EXPECT_THROW(promise.set_exception(std::out_of_range("ruh roh again")), future_ready);
    EXPECT_THROW(promise.set_exception(std::make_exception_ptr(std::out_of_range("ruh roh again"))), future_ready);
}