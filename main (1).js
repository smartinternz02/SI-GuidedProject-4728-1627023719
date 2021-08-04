document.getElementById('imageUpload').onchange = function(event){
    console.log('Ínside upload Image');
    var image = document.getElementById("img-preview");
    image.src = URL.createObjectURL(event.target.files[0]);
    image.style.display = 'block';
}
