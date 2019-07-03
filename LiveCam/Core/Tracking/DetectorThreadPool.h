#pragma once

#include <condition_variable>
#include <functional>
#include <vector>
#include <memory>
#include <thread>
#include <mutex>
#include <queue>

class DetectorThreadPool
{
public:
	using Task = std::function<void(void)>;

	explicit DetectorThreadPool(std::size_t numThreads)
	{
		_start(numThreads);
	}

	~DetectorThreadPool()
	{
		_stop();
	}

	void enqueue(Task task)
	{
		{
			std::unique_lock<std::mutex> lock(eventMutex);
			tasks.emplace(std::move(task));
		}

		eventVar.notify_one();
	}

private:
	std::vector<std::thread> threads;
	std::condition_variable eventVar;
	std::queue<Task> tasks;

	std::mutex eventMutex;
	bool quit = false;

	void _start(std::size_t numThreads)
	{
		for (auto i = 0u; i < numThreads; ++i)
		{
			threads.emplace_back([=] {
				while (true)
				{
					Task task;

					{
						std::unique_lock<std::mutex> lock(eventMutex);

						eventVar.wait(lock, [=] { return quit || !tasks.empty(); });

						if (quit && tasks.empty())
							break;

						task = std::move(tasks.front());
						tasks.pop();
					}

					//std::this_thread::sleep_for(std::chrono::milliseconds(100));

					task();
				}
			});
		}

		for (auto &thread : threads)
		{
			thread.detach();
		}
	}

	void _stop() noexcept
	{
		{
			std::unique_lock<std::mutex> lock(eventMutex);
			quit = true;
		}

		eventVar.notify_all();

		/*for (auto &thread : threads)
		{
			if (thread.joinable())
			{
				thread.join();
			}
		}*/
	}

};