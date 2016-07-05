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

int main(int argc, char **argv)
{
	//test();
	test_msg(argc, argv);
	return 0;
}