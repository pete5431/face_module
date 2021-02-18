// Merge objects into one export.
module.exports = Object.assign({},
  require("./face_recognizer"),
  require("./face_capture"),
);