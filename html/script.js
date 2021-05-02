import Webcam from 'webcam-easy';
const canvasElement = document.getElementById('canvas');
const webcam = new Webcam(webcamElement, 'user', canvasElement, snapSoundElement);
webcam.start()
  .then(result =>{
    console.log("webcam started");
  })
  .catch(err => {
    console.log(err);
});
let picture = webcam.snap();
document.querySelector('#download-photo').href = picture;
