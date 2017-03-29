
function getRandomImage(){
    httpGetAsync('/noise_filter/getRandomImage', function(new_id){
        console.log(new_id);
        $("#disp-image").attr("src", "normal/"+new_id);
        classify();
    })
}

function addNoise(){
    var info = $("#disp-image").attr("src");
    var n = info.indexOf("/");
    var image_id = info.substring(n + 1, info.length);
    $("#disp-image").attr("src", "noised/"+image_id);
    classify();
}

function classify(){
    var info = $("#disp-image").attr("src");
    httpGetAsync('predict/' + info, function(predict_res){
        var json = JSON.parse(predict_res);
        var results = "<p>The predicted label is " + json["label_predicted"].toString() + "</p>";
        results += "<p>The true label is " + json["label_true"].toString() + "</p>";
        var preds = json["prediction"];
        console.log(preds);
        var preds_args = json["prediction_args"];
        var pred_str = "";
        for (var i=0; i < 3; i+=1) {
            pred_str += '[' + preds_args[i] + ', ' + Number(preds[preds_args[i]]).toString() + ']; ';
        }
        results += "<p>" + pred_str + "</p>";
        $("#image-prediction").html(results);
    })
}

function httpGetAsync(theUrl, callback)
{
    var xmlHttp = new XMLHttpRequest();
    xmlHttp.onreadystatechange = function() {
        if (xmlHttp.readyState == 4 && xmlHttp.status == 200)
            callback(xmlHttp.responseText);
    }
    xmlHttp.open("GET", theUrl, true); // true for asynchronous
    xmlHttp.send(null);
}

$(document).ready(function() {
    classify();
});