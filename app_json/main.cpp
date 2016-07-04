//vs2015支持很好
// https://github.com/nlohmann/json
#include "json.hpp"
#include <fstream>
#include <sstream>
#include <mpi.h>


using json = nlohmann::json;
using namespace std;

void test_json()
{
	// a JSON text
	auto text = R"(
    {
        "Image": {
            "Width":  800,
            "Height": 600,
            "Title":  "View from 15th Floor",
            "Thumbnail": {
                "Url":    "http://www.example.com/image/481989943",
                "Height": 125,
                "Width":  100
            },
            "Animated" : false,
            "IDs": [116, 943, 234, 38793]
        }
    }
    )";

	// fill a stream with JSON text
	std::stringstream ss;
	ss << text;

	// create JSON from stream
	json j_complete(ss);

	std::cout << "here\n";
	std::cout << std::setw(4) << j_complete << "\n\n";
}

void test_geometry()
{
	try {

		json j_(ifstream("./geometry.json"));
		if (j_.is_null())
		{
			;
		}
		cout << std::setw(4) << j_ << "\n\n";
		cout << std::setw(4) << j_["geometry"] << "\n\n";
		cout << std::setw(4) << j_["geometry"]["p1"] << "\n\n";
		
		auto p1 = j_["geometry"]["p1"];
		auto p2 = j_["geometry"]["p2"];
		auto p3 = j_["geometry"]["p3"];
		auto p4 = j_["geometry"]["p4"];

		cout << p1["line"].is_null()<< endl;
		cout << p1["line"].is_number() << endl;
		cout << p1["line"] << endl;

		double line = p1["line"].get<double>();

		cout << line << endl;



	}
	catch (exception const &e){
		cerr << e.what() << endl;
	}
}

void test_input()
{
	try {

		json j_(ifstream("./input.json"));
		if (j_.is_null())
		{
			;
		}
		
		auto input = j_["input"];

		cout << input["prestk"] << endl;
		cout << input["type"] << endl;
		cout << input["files"] << endl;
		cout << input["info"] << endl;

	}
	catch (exception const &e) {
		cerr << e.what() << endl;
	}

}

void test_output()
{
	try {

		json j_(ifstream("./output.json"));
		if (j_.is_null())
		{
			;
		}

		auto output = j_["output"];

		cout << output["stacked"] << endl;
		cout << output["path"] << endl;
		cout << output["type"] << endl;
		cout << output["regions"] << endl;
		cout << output["segments"] << endl;
	}
	catch (exception const &e) {
		cerr << e.what() << endl;
	}

}

void test_velocity()
{
	try {

		json j_(ifstream("./velocity.json"));
		if (j_.is_null())
		{
			;
		}

		auto velocity = j_["velocity"];

		cout << velocity["type"] << endl;
		cout << velocity["file0"] << endl;
		cout << velocity["file1"] << endl;
		cout << velocity["scale0"] << endl;
		cout << velocity["scale1"] << endl;
		cout << velocity["geometry"] << endl;

	}
	catch (exception const &e) {
		cerr << e.what() << endl;
	}

}


void test_checkpoint()
{
	try {

		json j_(ifstream("./checkpoint.json"));
		if (j_.is_null())
		{
			;
		}

		auto ckpt = j_["checkpoint"];

		cout << ckpt["dir"] << endl;
		cout << ckpt["time"] << endl;
		cout << ckpt["itrace"] << endl;
		cout << ckpt["nckpt"] << endl;
	}
	catch (exception const &e) {
		cerr << e.what() << endl;
	}

}

void test_xmigt()
{
	try {

		json j_(ifstream("./xmigt.json"));
		if (j_.is_null())
		{
			;
		}

		auto xmigt = j_["xmigt"];

		cout << xmigt["dimension"] << endl;
		cout << xmigt["weight"] << endl;
		cout << xmigt["restarted"] << endl;
		cout << xmigt["tmpdir"] << endl;
		cout << xmigt["aperture"] << endl;
		cout << xmigt["alias"] << endl;

	}
	catch (exception const &e) {
		cerr << e.what() << endl;
	}

}

void test()
{
	test_geometry();
	test_input();
	test_output();
	test_velocity();
	test_checkpoint();
	test_xmigt();
}

/*
测试多个MPI 节点，并发读取一个本地数据，
本地数据是一个大文件的一部分，期间所有的数据会通过MPI通信，
在所有节点之间遍历一次(仅遍历一次):

如果有N个节点，相当于有N个硬盘的RAID0, 读盘的同时，网络也在
交换数据！ 平均IO所用时间：t=O(T/N)， T为一个节点读取整个大文件的时间

通过nonblock IO, 计算会和IO overlapped, IO时间会被掩盖

*/

struct MsgHD
{
    uint16_t  flags;
	uint16_t  count;
	int nrec;
	char addr[0];
};

typedef float rec_t[1024];
const static int NREC_PER_IO = 1024;

void test_msg(int argc, char **argv)
{
	cout << sizeof(MsgHD) << endl;

	size_t msg_len = sizeof(MsgHD) + sizeof(rec_t)*NREC_PER_IO;
	cout << "msg_len="<< msg_len << endl;

	MsgHD *disk_bufs[2];
	MsgHD *recv_bufs[2];

	char *buf_base=new char[4*msg_len];
	disk_bufs[0] = reinterpret_cast<MsgHD*>(buf_base);
	disk_bufs[1] = reinterpret_cast<MsgHD*>(buf_base+msg_len);
	recv_bufs[0] = reinterpret_cast<MsgHD*>(buf_base+2*msg_len);
	recv_bufs[1] = reinterpret_cast<MsgHD*>(buf_base+3*msg_len);

	cout << disk_bufs[0] << endl;
	cout << disk_bufs[1] << endl;
	cout << recv_bufs[0] << endl;
	cout << recv_bufs[1] << endl;

	int disk_current = 0;
	int recv_current = 0;
	
	MsgHD *compute_and_send_buf = 0;
	MPI_Request reqs[2];

	/*********************************/
	
	//初始化工作

	int myid, nnode;
	int prev, next;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	MPI_Comm_size(MPI_COMM_WORLD, &nnode);
	prev = myid - 1;
	next = myid + 1;
	
	prev = (prev + nnode) % nnode;
	next = next %nnode;

	char *filename = "file.txt";
	MPI_File fh;
	MPI_File_open(MPI_COMM_SELF, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);

	int nbytes = sizeof(rec_t)*NREC_PER_IO;
	
	//异步磁盘读数据
	MPI_File_iread(fh, disk_bufs[disk_current]->addr, nbytes, MPI_BYTE, &reqs[0]);
	//异步网络接收数据
	MPI_Irecv(recv_bufs[recv_current], msg_len, MPI_BYTE, prev, MPI_ANY_TAG, MPI_COMM_WORLD, &reqs[1]);

	while (0)
	{
		int which=-1;
		MPI_Status status;

		MPI_Waitany(2, reqs, &which, &status);
		if (which == 0) //处理磁盘IO
		{
			compute_and_send_buf = disk_bufs[disk_current];
			//compute_and_send_buf->nrec = 0;
			//compute_and_send_buf->flags=0;
			//compute_and_send_buf->count=0;
			disk_current = (disk_current + 1) % 2;

			//异步磁盘读数据
			MPI_File_iread(fh, disk_bufs[disk_current]->addr, nbytes, MPI_BYTE, &reqs[0]);

		}
		else if (which == 1) //处理网络数据
		{
			compute_and_send_buf = recv_bufs[recv_current];
			//compute_and_send_buf->count++;
			//compute_and_send_buf->nrec=0;
			//compute_and_send_buf->flags=0;

			recv_current = (recv_current + 1) % 2;

			//异步网络接收数据
			MPI_Irecv(recv_bufs[recv_current], msg_len, MPI_BYTE, prev, MPI_ANY_TAG, MPI_COMM_WORLD, &reqs[1]);
		}

		MPI_Status send_status;
		MPI_Request send_req;
		if (compute_and_send_buf->count != nnode)
		{
			//异步发送数据给NEXT节点
			MPI_Isend(compute_and_send_buf, msg_len, MPI_BYTE, next, MPI_ANY_TAG, MPI_COMM_WORLD, &send_req);
		}

		//调用计算函数
		//compute(compute_and_send_buf);

		
		if (compute_and_send_buf->count != nnode)
		{
			MPI_Wait(&send_req, &send_status);
			//wait(&send_req);
		}
	}
	MPI_File_close(&fh);
	delete[]buf_base;
}

int main(int argc, char **argv)
{
	//test();
	test_msg(argc, argv);
	return 0;
}