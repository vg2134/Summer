/**
 * 弹出式提示框，默认1.2秒自动消失
 * @param message 提示信息
 * @param style 提示样式，有alert-success、alert-danger、alert-warning、alert-info
 * @param time 消失时间
 */
var prompt = function (message, style, time) {
    style = (style === undefined) ? 'alert-success' : style;
    time = (time === undefined) ? 1200 : time;
    $('<div id="promptModal">')
        .appendTo('body')
        .addClass('alert ' + style)
        .css({
            "display": "block",
            "z-index": 99999,
            "left": ($(document.body).outerWidth(true) - 120) / 2,
            "top": ($(window).height() - 45) / 2,
            "position": "absolute",
            "padding": "20px",
            "border-radius": "5px"
        })
        .html(message)
        .show()
        .delay(5000)
        .fadeOut(10, function () {
            $('#promptModal').remove();
        });
};

// 成功提示
var success_prompt = function (time) {
    $.ajax({
            url: 'http://47.101.192.147:8000/hwtgenml/connect_collection_api/',
            data: {
                "data": 1
            },
            contentType: false,
            processData: false,
            xhrFields: {withCredentials: true},
            success: function () {
                prompt("connect successfully", 'alert-success', time);
                setTimeout(window.location.reload, 5000);
            },
            error: function () {
                prompt("connect fail", 'alert-success', time);
                setTimeout(window.location.reload, 5000);
            },

        }
    )

};

// 失败提示
var fail_prompt = function (message, time) {
    prompt(message, 'alert-danger', time);
};

// 提醒
var warning_prompt = function (message, time) {
    prompt(message, 'alert-warning', time);
};

// 信息提示
var info_prompt = function (message, time) {
    prompt(message, 'alert-info', time);
};

// 信息提示
var alert_prompt = function (message, time) {
    prompt(message, 'alert-pormpt', time);
};