
/*
$(document).ready(function(){

    var promise = $.ajax({ timeout: 1000 });

    promise.done(function(jqXHR, textStatus) {
        if(textStatus==="timeout") {
            url: 'localhost:5000/table',
            success: function(data) {
                var table = document.getElementById("myTable");
                data['people'].forEach(function(elem){
                    console.log(elem);
                });
            } 
        }
    });

   /* $.ajax({
        url: '/api/info',
        success: function(data) {
            console.log('get info');
            $('#info').html(JSON.stringify(data, null, '   '));
            $('#description').html(data['description']);
        }
    });


    $.ajax({
            url : 'table',
            success: function(data) {
                var table = document.getElementById("myTable");
                data['people'].forEach(function(elem){
                    console.log(elem)
                });
            }
        });
*/
    
$(document).ready(function(){
    $('#calc').click(function() {
        console.log("Hello1");
        $.ajax({
            url: 'localhost:5000/table',
            success: function(data) {
                    console.log("Hello");
                    /*var table = document.getElementById("myTable");
                    data['people'].forEach(function(elem){
                        console.log(elem);
                    });*/
            } 
        });
    });
});