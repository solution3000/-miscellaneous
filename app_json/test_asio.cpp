/*
	模仿Asio Echo Server方式，开发一问一答服务器
*/

#include <cstdlib>
#include <iostream>
#include <boost/bind.hpp>
#include <boost/asio.hpp>
#include <boost/thread.hpp>
#include <boost/random.hpp>
#include <boost/random/random_device.hpp>

using boost::asio::ip::tcp;
boost::random::mt19937 gen;
boost::uniform_int<> real(1, 10999);

struct IndexScore
{
	int index;
	float score;
};

namespace  Buffers {

	//CPU Buffers
	static boost::thread_specific_ptr<float> my_fea;
	static boost::thread_specific_ptr<IndexScore> my_result;

	//GPU Buffers
	static boost::thread_specific_ptr<float> d_my_fea;
	static boost::thread_specific_ptr<IndexScore> d_my_result;
};

namespace constants
{
	enum { N_FEA = 90 };
	enum { N_RESULT = 100 };
};


//客户端TCP链接对话
class session
{
public:
	session(boost::asio::io_service& io_service)
		: socket_(io_service)
	{
	}

	tcp::socket& socket()
	{
		return socket_;
	}

	void start()
	{
		//发起异步读
		//特征码数据BUFFER
		boost::asio::async_read(socket_, boost::asio::buffer((char*)fea_, constants::N_FEA * sizeof(fea_[0])),
			boost::bind(&session::handle_read, this,
				boost::asio::placeholders::error,
				boost::asio::placeholders::bytes_transferred));
	}

private:

	//读入已经准备好的数据
	void handle_read(const boost::system::error_code& error,
		size_t bytes_transferred)
	{
		if (!error)
		{
			init_buffer();

			std::cerr << boost::this_thread::get_id()
				<< ", " << Buffers::my_fea.get()
				<< ", " << Buffers::my_result.get()
				<< std::endl;

			
			/*
			此处sleep模拟计算场景
			*/
			//根据特征码，进行1:N搜脸计算，result_是结果！
			for (int i = 0; i < constants::N_RESULT; i++)
			{
				result_[i].index = constants::N_RESULT - i;
				result_[i].score = sqrt(i);
			}
			result_[0].score = fea_[0];
			boost::this_thread::sleep(boost::posix_time::milliseconds(real(gen)));


			//返回计算结果
			boost::asio::async_write(socket_,
				boost::asio::buffer((char*)result_, constants::N_RESULT * sizeof(result_[0])),
				boost::bind(&session::handle_write, this,
					boost::asio::placeholders::error));
		}
		else
		{
			delete this;
		}
	}

	//处理写完
	void handle_write(const boost::system::error_code& error)
	{
		//销毁本次对话，socket_会自动关闭
		delete this;
	}
private:
	tcp::socket socket_;
private:
	float fea_[constants::N_FEA];
	IndexScore result_[constants::N_RESULT];
private:
	void init_buffer()
	{
		if (!Buffers::my_fea.get())
		{
			Buffers::my_fea.reset(new float[constants::N_FEA]);
		}

		if (!Buffers::my_result.get())
		{
			Buffers::my_result.reset(new IndexScore[constants::N_RESULT]);
		}

		return;
	}
};

//TCP服务器
class server
{
public:
	server(boost::asio::io_service& io_service, short port)
		: io_service_(io_service),
		acceptor_(io_service, tcp::endpoint(tcp::v4(), port))
	{
		start_accept();
	}
	//使用线程池
	void run()
	{
		// Create a pool of threads to run all of the io_services.
		std::vector<boost::shared_ptr<boost::thread> > threads;
		for (std::size_t i = 0; i < thread_pool_size_; ++i)
		{
			boost::shared_ptr<boost::thread> thread(new boost::thread(
				boost::bind(&boost::asio::io_service::run, &io_service_)));
			threads.push_back(thread);
		}

		// Wait for all threads in the pool to exit.
		for (std::size_t i = 0; i < threads.size(); ++i)
			threads[i]->join();
	}
private:
	void start_accept()
	{
		session* new_session = new session(io_service_);
		acceptor_.async_accept(new_session->socket(),
			boost::bind(&server::handle_accept, this, new_session,
				boost::asio::placeholders::error));
	}
private:
	void handle_accept(session* new_session,
		const boost::system::error_code& error)
	{
		if (!error)
		{
			new_session->start();
		}
		else
		{
			delete new_session;
		}

		start_accept();
	}
private:
	boost::asio::io_service& io_service_;
private:
	tcp::acceptor acceptor_;
private:
	const std::size_t thread_pool_size_ = 4;
};

#if 0

int main(int argc, char* argv[])
{
	try
	{
		if (argc != 2)
		{
			std::cerr << "Usage: async_tcp_echo_server <port>\n";
			return 1;
		}

		boost::asio::io_service io_service;

		using namespace std;
		server s(io_service, atoi(argv[1]));

		//io_service.run();
		//使用线程池
		s.run();
	}
	catch (std::exception& e)
	{
		std::cerr << "Exception: " << e.what() << "\n";
	}

	return 0;
}

#endif

