function FaceCapture(constraints, video){
  this.constraints = constraints;
  this.video = video;
}

FaceCapture.prototype.startStream = async function startStream(){
  let videoStream = null;
  try {
    videoStream = await navigator.mediaDevices.getUserMedia(this.constraints);
    // Set the src of the video as the video stream.
    this.video.srcObject = videoStream;
    // Play the video.
    this.video.play();
  } catch(err) {
    console.log(err);
  }
}

FaceCapture.prototype.drawCanvas = function drawCanvas(canvas, video){
  // Set canvas width and height to be the same as video.
  canvas.width = this.constraints.video.width;
  canvas.height = this.constraints.video.height;
  // Clear the canvas.
  canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
  // Draw the video onto the canvas.
  canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
}

FaceCapture.prototype.takePicture = function takePicture(){
  // Create new canvas element.
  const canvas = document.createElement("canvas");
  // Draw video onto canvas.
  this.drawCanvas(canvas, this.video);
  // Return data URL of the loaded image.
  return canvas.toDataURL('image/jpg');
}

module.exports = {FaceCapture: FaceCapture};
