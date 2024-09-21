import {
  FilesetResolver,
  ObjectDetector,
  ObjectDetectorResult,
} from "@mediapipe/tasks-vision";

export const initializeObjectDetector = async (
  videoRef: React.RefObject<HTMLVideoElement>,
  canvasRef: React.RefObject<HTMLCanvasElement>,
  setDetectedObjects: React.Dispatch<React.SetStateAction<string[]>>,
  detectionTolerance: number // Pass detection tolerance as a parameter
) => {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
  );

  const objectDetector = await ObjectDetector.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: `https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite2/float32/1/efficientdet_lite2.tflite`,
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    scoreThreshold: detectionTolerance, // Use the passed detectionTolerance
    categoryAllowlist: ["person"],
  });

  const video = videoRef.current;
  const canvas = canvasRef.current;

  if (video && canvas) {
    const vWidth = window.innerWidth;
    const vHeight = window.innerHeight;

    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: vWidth, height: vHeight },
    });

    video.srcObject = stream;

    video.onloadeddata = () => {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      video.play();
      renderLoop(objectDetector, videoRef, canvasRef, setDetectedObjects);
    };
  }
};


let lastVideoTime = -1;

const renderLoop = async (
  objectDetector: ObjectDetector,
  videoRef: React.RefObject<HTMLVideoElement>,
  canvasRef: React.RefObject<HTMLCanvasElement>,
  setDetectedObjects: React.Dispatch<React.SetStateAction<string[]>>
) => {
  const video = videoRef.current;
  const canvasCtx = canvasRef.current?.getContext("2d");

  if (!video || !canvasCtx) {
    console.error("Video or canvas context not found.");
    return;
  }

  const render = async () => {
    if (!video || video.paused || video.ended) {
      console.log("Video is not ready or has ended.");
      return;
    }

    if (video.currentTime !== lastVideoTime) {
      lastVideoTime = video.currentTime;

      try {
        const results: ObjectDetectorResult = await objectDetector.detectForVideo(
          video,
          performance.now()
        );

        if (results.detections.length === 0) {
          console.log("No detections found.");
        } else {
          processResults(
            results,
            canvasCtx,
            video.videoWidth,
            video.videoHeight,
            canvasRef.current!.width,
            canvasRef.current!.height
          );
          setDetectedObjects(
            results.detections.map(
              (d) =>
                `${d.categories[0].categoryName} (${Math.round(
                  d.categories[0].score * 100
                )}%)`
            )
          );
        }
      } catch (error) {
        console.error("Error during object detection:", error);
      }
    }

    requestAnimationFrame(render);
  };

  requestAnimationFrame(render);
};

const processResults = (
  results: ObjectDetectorResult,
  canvasCtx: CanvasRenderingContext2D | null,
  originalVideoWidth: number,
  originalVideoHeight: number,
  canvasWidth: number,
  canvasHeight: number
) => {
  if (!canvasCtx) return;

  // Clear the canvas before drawing
  canvasCtx.clearRect(0, 0, canvasCtx.canvas.width, canvasCtx.canvas.height);

  // Calculate the scaling factors based on the original video dimensions and the canvas dimensions
  const scaleX = canvasWidth / originalVideoWidth;
  const scaleY = canvasHeight / originalVideoHeight;

  // Draw each bounding box based on the scaled coordinates
  results.detections.forEach((detection) => {
    const bbox = detection.boundingBox;

    if (bbox) {

      // Scale the bounding box coordinates
      const x = bbox.originX * scaleX;
      const y = bbox.originY * scaleY;
      const width = bbox.width * scaleX;
      const height = bbox.height * scaleY;

      // Draw the bounding box around the detected object
      canvasCtx.strokeStyle = "yellow";
      canvasCtx.lineWidth = 4;
      canvasCtx.strokeRect(x, y, width, height);
    }
  });
};
