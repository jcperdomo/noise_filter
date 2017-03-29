
function getRandomImage(){
    httpGetAsync('/noise_filter/getRandomImage', function(new_id){
        console.log(new_id);
        $("#disp-image").attr("src", "normal/"+new_id);
    })
}

function addNoise(){
    var info = $("#disp-image").attr("src");
    var n = info.indexOf("/");
    var image_id = info.substring(n + 1, info.length);
    $("#disp-image").attr("src", "noised/"+image_id);
}

function classify(){
    var info = $("#disp-image").attr("src");
    httpGetAsync('predict/' + info, function(predict_res){
        var json = JSON.parse(predict_res);
        var results = "The predicted label is " + json.label.toString()
        $("#image-prediction").attr("text", results);
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
