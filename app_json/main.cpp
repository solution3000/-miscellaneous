//vs2015支持很好
// https://github.com/nlohmann/json
#include "json.hpp"
#include <fstream>
#include <sstream>

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

struct MsgHD
{
    uint16_t  flags;
	uint16_t  count;
	int nrec;
	char addr[0];
};

typedef float rec_t[1024];
const static int NREC_PER_IO = 1024;

void test_msg()
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

	//issue disk_aio(disk_bufs[disk_current].addr, &reqs[0]);
	//issue recv_aio(recv_bufs[recv_current], &reqs[1]);
	while (0)
	{
		int which=0;
		//waitany(2, reqs, which, status);
		if (which == 0) //disk io
		{
			compute_and_send_buf = disk_bufs[disk_current];
			//compute_and_send_buf->nrec = 0;
			//compute_and_send_buf->flags=0;
			//compute_and_send_buf->count=0;
			disk_current = (disk_current + 1) % 2;
			//issue disk_aio(disk_bufs[disk_current].addr, &reqs[0]);

		}
		else if (which == 1) //net io
		{
			compute_and_send_buf = recv_bufs[recv_current];
			//compute_and_send_buf->count++;
			//compute_and_send_buf->nrec=0;
			//compute_and_send_buf->flags=0;

			recv_current = (recv_current + 1) % 2;
			//issue recv_aio(recv_bufs[recv_current], &reqs[1]);
		}
		int nnode = 4;
		if (compute_and_send_buf->count != nnode)
		{
			//issue send_aio(compute_and_send_buf, &send_req);
		}
		
		//compute(compute_and_send_buf);

		
		if (compute_and_send_buf->count != nnode)
		{
			//wait(&send_req);
		}
	}
	
	delete[]buf_base;
}

int main(int argc, char **argv)
{
	//test();
	test_msg();
	return 0;
}