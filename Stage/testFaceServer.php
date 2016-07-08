<?php

$ip = '192.168.1.181';
$port = 80;
// 返回结果中, 单个数据的长度
$unitLength = 8;

//特征码文件路径
$fea_file = "/Users/liliangzhu/Desktop/f_0001.fea";

while (1) {
	$socket = socket_create(AF_INET, SOCK_STREAM, SOL_TCP);
	$con=socket_connect($socket, $ip, $port);

	if(!$con){
		echo "socket connect error";
		socket_close($socket);
		exit;
	}

	
	$content = file_get_contents($fea_file);

	$w = socket_write($socket,$content);

	$out=@socket_read($socket,2048);
	// var_dump($out, strlen($out));
	$format = 'iindex/fscore';


	$result = array();
	for($i=0; $i<strlen($out); $i+=$unitLength)
	{
		$c = substr($out, $i, $unitLength);
		$result[] = unpack($format, $c);
	}
	print_r($result[0]);
	echo time()."\n";
	file_put_contents("/tmp/testServer.log", $result[0]["score"]."\n", FILE_APPEND);
	socket_close($socket);
}