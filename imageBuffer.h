#ifndef IMAGEBUFFER_H
#define IMAGEBUFFER_H

#include <opencv2/opencv.hpp>
#include <mutex>
#include <condition_variable>
#include <queue>


template<typename T>
class ConsumerProducerQueue
{

public:
    ConsumerProducerQueue(int mxsz,bool dropFrame) :
            maxSize(mxsz),dropFrame(dropFrame)
    { }

    void add(T request)
    {
        std::unique_lock<std::mutex> lock(mutex);
        if(dropFrame && isFull())
        {
            lock.unlock();
                return;
        }
        else {
            cond.wait(lock, [this]() { return !isFull(); });
            cpq.push(request);
            //lock.unlock();
            cond.notify_all();
        }
    }

    void consume(T &request)
    {
        std::unique_lock<std::mutex> lock(mutex);
        cond.wait(lock, [this]()
        { return !isEmpty(); });
        request = cpq.front();
        cpq.pop();
        //lock.unlock();
        cond.notify_all();

    }

    bool isFull() const
    {
        return cpq.size() >= maxSize;
    }

    bool isEmpty() const
    {
        return cpq.size() == 0;
    }

    int length() const
    {
        return cpq.size();
    }

    void clear()
    {
        std::unique_lock<std::mutex> lock(mutex);
        while (!isEmpty())
        {
            cpq.pop();
        }
        lock.unlock();
        cond.notify_all();
    }

private:
    std::condition_variable cond;
    std::mutex mutex;
    std::queue<T> cpq;
    int maxSize;
    bool dropFrame;
};



#endif
