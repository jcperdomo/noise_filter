
function getRandomImage(){
    httpGetAsync('/noise_filter/getRandomImage', function(new_id){
        console.log(new_id);
        $("#disp-image").attr("src", "normal/"+new_id);
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
