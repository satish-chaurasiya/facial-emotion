function openCvReady() {
  let imgElement = document.getElementById("imageSrc");
  let inputElement = document.getElementById("fileInput");
  inputElement.addEventListener(
    "change",
    (e) => {
      imgElement.src = URL.createObjectURL(e.target.files[0]);
    },
    false
  );

  function detectFace(img) {
    let utils = new Utils("errorMessage"); //use utils class
    // load pre-trained classifiers
    let faceCascadeFile =
      "http://127.0.0.1:8000/static/js/haarcascade_frontalface_default.xml"; // path to xml

    let path = 'haarcascade_frontalface_default.xml'

    // use createFileFromUrl to "pre-build" the xml
    utils.createFileFromUrl(path, faceCascadeFile, () => {
      let src = cv.imread(img);
      let gray = new cv.Mat();
      cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY, 0);
      let faces = new cv.RectVector();
      let faceCascade = new cv.CascadeClassifier(); // initialize classifier
      faceCascade.load(path);

      let msize = new cv.Size(0, 0);
      faceCascade.detectMultiScale(gray, faces, 1.1, 3, 0, msize, msize);
      for (let i = 0; i < faces.size(); ++i) {
        let roiGray = gray.roi(faces.get(i));
        let roiSrc = src.roi(faces.get(i));
        let point1 = new cv.Point(faces.get(i).x, faces.get(i).y);
        let point2 = new cv.Point(
          faces.get(i).x + faces.get(i).width,
          faces.get(i).y + faces.get(i).height
        );
        cv.rectangle(src, point1, point2, [255, 0, 0, 255]);
        roiGray.delete();
        roiSrc.delete();
      }
      cv.imshow("canvasOutput", src);
      src.delete();
      gray.delete();
      faceCascade.delete();
      faces.delete();
    });
  }

  imgElement.onload = function () {
    detectFace(imgElement);
  };
}
