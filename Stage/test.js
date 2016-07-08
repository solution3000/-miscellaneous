$('#sendVerifySmsButton').sms({
    //laravel csrf token
    token           : "sEvgRnf1hoO6bM9DNiy8HsCjXaXebjdcemGuUXXQ",
    //定义如何获取mobile的值
    mobile_selector : 'input[name=telephone]',
    //手机号的检测规则
    mobile_rule     : 'mobile_required',
    //语音验证
    voice           : false,
    //请求间隔时间
    interval        : 1 
});

function timed()
{
   	$('#sendVerifySmsButton').click();
	setTimeout("timed()", 2000);
}


$('#sendVerifySmsButton').sms({
    //laravel csrf token
    token           : "sEvgRnf1hoO6bM9DNiy8HsCjXaXebjdcemGuUXXQ",
    //定义如何获取mobile的值
    mobile_selector : 'input[name=telephone]',
    //手机号的检测规则
    mobile_rule     : 'mobile_required',
    //语音验证
    voice           : false,
    //请求间隔时间
    interval        : 1 
});