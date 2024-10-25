#pragma once
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <queue>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <unistd.h>

#include <condition_variable>
#include <future>

#include <chrono>
#include <ctime>
#include <iomanip>

#include <algorithm>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include "Logger.h"

template <typename T>
void check(T result, char const* const func, const char* const file, int const line)
{
    if (result)
    {
        logger->error("CUDA error at {}:{} code={} \"{}\"", file, line, static_cast<unsigned int>(result), func);
        // Make sure we call CUDA Device Reset before exiting
        exit(EXIT_FAILURE);
    }
}
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

#define kSecondsToNanos (1000ULL * 1000ULL * 1000ULL)
inline uint64_t GetCpuTimestamp()
{
    timespec m_cputs;
    clock_gettime(CLOCK_REALTIME, &m_cputs);
    return static_cast<uint64_t>(m_cputs.tv_sec) * kSecondsToNanos + static_cast<uint64_t>(m_cputs.tv_nsec);
}

inline void safeCudaFree(void* ptr)
{
    if (ptr == 0)
        return;
    cudaFree(ptr);
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        logger->error("[{} {}]: safeCudaFree CUDA error: ({}) {}", __FUNCTION__, __LINE__, static_cast<int>(err), cudaGetErrorString(err));
    }
}

inline void* safeCudaMalloc(size_t memSize)
{
    void* deviceMem;
    checkCudaErrors(cudaMalloc(&deviceMem, memSize));
    if (deviceMem == nullptr)
    {
        logger->error("[{} {}]: Out of memory", __FUNCTION__, __LINE__);
        exit(1);
    }
    return deviceMem;
}

class CudaCtxPush final
{
public:
    explicit CudaCtxPush(CUcontext ctx) { cuCtxPushCurrent(ctx); }
    ~CudaCtxPush() { cuCtxPopCurrent(nullptr); }
};

class CudaStrSync final
{
    CUstream str;

public:
    explicit CudaStrSync(CUstream stream) { str = stream; }
    ~CudaStrSync() { cuStreamSynchronize(str); }
};

enum Pixel_Format
{
    UNDEFINED      = 0,
    Y              = 1,
    RGB            = 2,
    NV12           = 3,
    YUV420         = 4,
    RGB_PLANAR     = 5,
    BGR            = 6,
    YCBCR          = 7,
    YUV444         = 8,
    RGB_32F        = 9,
    RGB_32F_PLANAR = 10,
    YUV422         = 11,
    P10            = 12,
    P12            = 13,
    YUV444_10bit   = 14,
    YUV420_10bit   = 15,
    NV12_Planar    = 16,
    GRAY12         = 17,
    BGR_PLANAR     = 18,
};

class TaskThread
{
public:
    TaskThread()                  = default;
    TaskThread(const TaskThread&) = delete;
    TaskThread& operator=(const TaskThread& other) = delete;

    TaskThread(std::thread&& thread)
        : t(std::move(thread))
    {
        finsh.store(false);
    }

    TaskThread(TaskThread&& thread)
        : t(std::move(thread.t))
    {
        finsh.store(false);
    }

    TaskThread& operator=(TaskThread&& other)
    {
        finsh.store(false);
        t = std::move(other.t);
        return *this;
    }

    ~TaskThread()
    {
        finsh.store(true);
        join();
    }

    void join()
    {
        if (t.joinable())
        {
            t.join();
        }
    }

    bool get_status() { return finsh.load(); }
    void set_status(bool status) { finsh.store(status); }

    std::thread::id threadId;

private:
    std::atomic<bool> finsh = false;
    std::thread       t;
};

template <typename T>
class ProcessQueue
{
public:
    ProcessQueue(size_t max_size)
        : max_size_(max_size)
        , count_(0)
        , drop_(0)
    {
    }

    ~ProcessQueue() { clear(); }

    // 添加元素到队列
    void put(const T& item)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        if (queue_.size() >= max_size_)
        {
            queue_.pop();  // 如果队列已满，删除最前面的元素
            drop_++;
        }
        queue_.push(item);
        count_++;
        lock.unlock();
        cond_var_.notify_all();
        return;
    }

    // 从队列中获取单个元素
    std::future<T> get()
    {
        std::promise<T>              promise;
        std::unique_lock<std::mutex> lock(mutex_);
        if (queue_.empty())
        {
            pending_single_gets_.emplace(std::move(promise));
            return pending_single_gets_.back().get_future();
        }
        else
        {
            T item = queue_.front();
            queue_.pop();
            promise.set_value(item);
            return promise.get_future();
        }
    }

    // 从队列中获取单个元素，带超时时间
    std::future<T> get(std::chrono::milliseconds timeout)
    {
        std::promise<T>              promise;
        std::unique_lock<std::mutex> lock(mutex_);

        if (queue_.empty())
        {
            bool success = cond_var_.wait_for(lock, timeout, [this]() { return !queue_.empty(); });
            if (!success)
            {
                promise.set_value(T());
                return promise.get_future();
            }
        }

        T item = queue_.front();
        queue_.pop();
        promise.set_value(item);
        return promise.get_future();
    }

    // 从队列中获取多个元素，带超时时间
    std::future<std::vector<T>> get(size_t size, std::chrono::milliseconds timeout = std::chrono::milliseconds(1))
    {
        std::promise<std::vector<T>> promise;
        std::unique_lock<std::mutex> lock(mutex_);

        if (queue_.size() < size)
        {
            cond_var_.wait_for(lock, timeout, [this, size]() { return queue_.size() >= size; });
        }

        std::vector<T> items;
        while (!queue_.empty() && items.size() < size)
        {
            items.push_back(queue_.front());
            queue_.pop();
        }
        promise.set_value(std::move(items));
        return promise.get_future();
    }

    // 清空元素
    void clear()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        while (!queue_.empty())
        {
            queue_.pop();
        }
        // Clear all pending gets
        while (!pending_single_gets_.empty())
        {
            pending_single_gets_.front().set_value(T());
            pending_single_gets_.pop();
        }
        while (!pending_bulk_gets_.empty())
        {
            pending_bulk_gets_.front().first.set_value(std::vector<T>());
            pending_bulk_gets_.pop();
        }
    }

    // 获取当前队列中的元素数量
    size_t size() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

    // 获取当前队列中的元素数量
    size_t count() const { return count_.load(); }

    size_t drops_count() const { return drop_.load(); }

    void reset_count()
    {
        drop_.store(0);
        std::lock_guard<std::mutex> lock(mutex_);
        count_.store(queue_.size());
    }

private:
    mutable std::mutex          mutex_;                // 互斥锁，用于保护对队列的访问
    std::condition_variable     cond_var_;             // 条件变量，用于等待队列非空
    std::queue<T>               queue_;                // 存储数据的队列
    std::queue<std::promise<T>> pending_single_gets_;  // 存储等待获取单个元素的 promise
    std::queue<std::pair<std::promise<std::vector<T>>, size_t>>
                        pending_bulk_gets_;  // 存储等待获取多个元素的 promise 和 size
    size_t              max_size_;           // 队列的最大尺寸
    std::atomic<size_t> count_;              // 统计队列收集总数
    std::atomic<size_t> drop_;               // 统计队列收集总数
};

static bool createDirectory(const std::string& path)
{
    size_t lastSlashIdx = 0;
    size_t nextSlashIdx = path.find('/', lastSlashIdx + 1);

    while (nextSlashIdx != std::string::npos)
    {
        std::string subPath = path.substr(0, nextSlashIdx);
#ifdef _WIN32
        if (CreateDirectory(subPath.c_str(), NULL) || ERROR_ALREADY_EXISTS == GetLastError())
        {
            // Directory created successfully or already exists
        }
        else
        {
            return false;
        }
#else
        if (mkdir(subPath.c_str(), 0777) != 0 && errno != EEXIST)
        {
            return false;
        }
#endif

        lastSlashIdx = nextSlashIdx;
        nextSlashIdx = path.find('/', lastSlashIdx + 1);
    }

    return true;
}

inline int FolderExist(const std::string& filename)
{
    size_t last_slash_idx = filename.rfind('/');
    if (last_slash_idx != std::string::npos)
    {
        std::string folder_path = filename.substr(0, last_slash_idx) + "/";

        if (!createDirectory(folder_path))
        {
            logger->error("[{} {}]: Create folder failed: {}", __FUNCTION__, __LINE__, folder_path);
            return -1;  // 返回错误代码
        }
    }
    else
    {
        logger->error("[{} {}]: Image path is not valid: {}", __FUNCTION__, __LINE__, filename);
        return -2;
    }

    return 0;
}

inline std::string GetLocalTime(bool file = false)
{
    auto now = std::chrono::system_clock::now();

    std::time_t now_c = std::chrono::system_clock::to_time_t(now);

    struct tm parts;
    localtime_r(&now_c, &parts);

    char buffer[80];
    if (file)
        strftime(buffer, sizeof(buffer), "%Y-%m-%d/%H:%M:%S", &parts);
    else
        strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", &parts);
    // 返回格式化的时间字符串
    return std::string(buffer);
}

inline CUcontext Init(int dev)
{
    checkCudaErrors(cuInit(0));
    CUdevice nGpu = 0;
    CUdevice iGpu = dev;
    checkCudaErrors(cuDeviceGetCount(&nGpu));
    logger->info("[{} {}]: Number of GPUs: {}", __FUNCTION__, __LINE__, nGpu);

    CUdevice cuDevice = 0;
    checkCudaErrors(cuDeviceGet(&cuDevice, iGpu));
    logger->info("[{} {}]: cuDeviceGet is: {}", __FUNCTION__, __LINE__, cuDevice);

    char szDeviceName[80];
    checkCudaErrors(cuDeviceGetName(szDeviceName, sizeof(szDeviceName), cuDevice));
    logger->info("[{} {}]: GPU in use: {}", __FUNCTION__, __LINE__, szDeviceName);

    // create Context
    CUcontext cuContext;
    checkCudaErrors(cuCtxCreate(&cuContext, 0, cuDevice));
    // logger->info("{} {}, context:{}", __FUNCTION__, __LINE__, cuContext);

    return cuContext;
}

inline void writeCSV(const std::string& filename, const std::vector<std::vector<std::string>>& data)
{
    std::ofstream csvFile(filename, std::ios::app);
    if (!csvFile.is_open())
    {
        logger->error("Error opening file: {}", filename);
        return;
    }

    for (const auto& row : data)
    {
        for (auto it = row.begin(); it != row.end(); ++it)
        {
            csvFile << *it;
            if (std::next(it) != row.end())
            {
                csvFile << ",";  // Add comma if it's not the last element in the row
            }
        }
        csvFile << "\n";  // Move to the next line after each row
    }

    csvFile.close();
}

template <typename T>
std::string floatToFixedString(T value, int precision = 2)
{
    static_assert(std::is_floating_point<T>::value, "Only floating-point types are allowed.");

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << value;
    return oss.str();
}

inline void writeLOG(size_t       frames,
                     size_t       last_frame,
                     uint64_t     elapsedTime,
                     std::string& _input_source,
                     std::string& log_name)
{
    size_t      frame_10s  = frames - last_frame;
    float       _10s       = ((float)elapsedTime / kSecondsToNanos);
    float       qps        = ((float)frame_10s) / _10s;
    std::string _real_time = GetLocalTime();

    std::string                           str_qps = floatToFixedString(qps, 2);
    std::vector<std::vector<std::string>> data    = {
        {_real_time, _input_source, str_qps, std::to_string(frames)},
    };
    writeCSV(log_name, data);
}

inline std::string generateEnginePath(const std::string& onnx_path)
{
    size_t last_dot = onnx_path.find_last_of(".");
    if (last_dot == std::string::npos)
    {
        throw std::runtime_error("Invalid ONNX file path.");
    }
    return onnx_path.substr(0, last_dot) + ".engine";
}
