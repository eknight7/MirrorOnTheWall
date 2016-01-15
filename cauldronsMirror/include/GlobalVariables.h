// Variables for grogginess detection
#include <opencv2/core/core.hpp>
#define CAMERA_ID 0
#define DISPLAY 1
#define DEBUG 0

#define EYE_REGION_THRESH 40
#define MAX_VAL 255
#define DILATION_TYPE 2 // ELLIPSE
#define DILATION_SIZE 1

// NOTE: Full path to cascade XML files is REQUIRED to load them successfully
#define FACE_CASCADE "/Users/esphinx/opencv/data/haarcascades/haarcascade_frontalface_default.xml"
#define EYE_CASCADE "/Users/esphinx/opencv/data/haarcascades/haarcascade_eye.xml"

