ImageOps MCP Server
imageops-mcp-server is a FastMCP-based Python server designed for advanced image processing tasks. It provides a suite of tools for fetching, manipulating, and analyzing images from URLs or local file paths. The server leverages the YOLOv8 and YOLOv11 models for object detection and segmentation, OpenCV for edge detection, and PIL for image format conversion and cropping. The project is built to be extensible, efficient, and compatible with modern AI-driven workflows, making it suitable for integration with tools like Claude Desktop and Cursor.
Features

Image Fetching: Retrieve images from URLs or local files, with automatic compression for large images (<1MB).
Format Conversion: Convert images to JPEG, PNG, or GIF formats with optional saving to an output directory.
Object Detection: Perform object detection using YOLOv8, returning labels, confidence scores, and bounding box coordinates.
Object Segmentation: Apply YOLOv11 segmentation to overlay colored masks on detected objects with customizable transparency.
Image Cropping: Crop specific regions of an image using bounding box coordinates, compatible with YOLO detection results.
Canny Edge Detection: Detect edges in images using OpenCV's Canny algorithm with adjustable thresholds.
Logging: Comprehensive logging to both console and file with timestamped logs for debugging and monitoring.
Directory Management: Automatically creates necessary directories (data, Temp, output) for logs and processed images.
Integration Ready: Designed for use with AI tools like Claude Desktop and Cursor for seamless workflow integration.

Prerequisites
To set up and run the imageops-mcp-server, ensure you have the following installed:

Python: Version 3.13 or higher
uv: A modern Python package and project manager (recommended for dependency management)
Git: For cloning the repository
Claude Desktop (optional): For interacting with the server in an AI-driven workflow
Cursor (optional): For code editing and integration with AI assistants
Operating System: Compatible with Windows, macOS, or Linux

Setup Instructions
Follow these steps to set up the project using uv:

Clone the Repository:
git clone https://github.com/your-username/imageops-mcp-server.git
cd imageops-mcp-server


Install uv:If you don't have uv installed, follow the official installation instructions:
# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows (PowerShell)
iwr -useb https://astral.sh/uv/install.ps1 | iex


Create a Virtual Environment and Install Dependencies:Use uv to set up the project environment and install dependencies specified in project.toml:
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install .

This command reads the project.toml file and installs dependencies like fastmcp, httpx, ultralytics, opencv-contrib-python, pillow, and others.

Verify Directory Structure:The server automatically creates the following directories if they don't exist:

./data: Stores log files (e.g., DD-MM-YY.log).
./Temp: Temporary storage for intermediate files.
./output: Stores processed images (e.g., converted, cropped, segmented, or edge-detected images).

Ensure you have write permissions in the project directory.


Running the Server
To start the FastMCP server, run the main script:
python image_mcp_server.py

The server runs using the stdio transport, allowing it to process requests from compatible clients, such as Claude Desktop or custom scripts.
Testing with mcp dev
The mcp dev command (part of the mcp CLI) allows you to test the server's tools interactively. Follow these steps:

Ensure Dependencies:Confirm that mcp[cli] is installed (included in project.toml).

Run mcp dev:In the project directory, activate the virtual environment and start the development tool:
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
mcp dev


Test Tools:The mcp dev interface allows you to call the server's tools directly. For example:

Fetch Image:fetch_image "https://ultralytics.com/images/bus.jpg"

Returns an Image object or error message if the fetch fails.
Get Image Dimensions and Save:get_image_dimensions_and_save "https://ultralytics.com/images/bus.jpg"

Returns a dictionary with dimensions and saved path.
Convert Image Format:convert_image_format "https://ultralytics.com/images/bus.jpg" "png" true

Converts the image to PNG and saves it.
Detect Objects:detect_objects_yolo "https://ultralytics.com/images/bus.jpg" true

Performs YOLOv8 object detection and saves the annotated image.
Crop Object:crop_the_object "https://ultralytics.com/images/bus.jpg" 100 150 200 300 true

Crops the specified region and saves it.
Segment Objects:segment_objects_yolo "https://ultralytics.com/images/bus.jpg" 0.5 true

Performs YOLOv11 segmentation with 0.5 transparency.
Canny Edge Detection:canny_edge_detection "https://ultralytics.com/images/bus.jpg" 100 200 true

Applies Canny edge detection with specified thresholds.


Check Outputs:

Logs are saved in the ./data directory (e.g., 09-07-25.log).
Processed images are saved in the ./output directory with timestamped filenames (e.g., image_2025-07-09_171823.jpg).



Using with Claude Desktop
To integrate the server with Claude Desktop for AI-driven image processing:

Start the Server:Ensure the server is running:
python image_mcp_server.py


Configure Claude Desktop:

In Claude Desktop, configure the FastMCP client to connect to the server via the stdio transport.
Use Claude's interface to send requests to the server tools (e.g., fetch_image, detect_objects_yolo).
Example request in Claude Desktop:{
  "tool": "detect_objects_yolo",
  "args": {
    "image_source": "https://ultralytics.com/images/bus.jpg",
    "save": true
  }
}




View Results:

Claude Desktop will display the JSON response, including detection details or image data.
Check the ./output directory for saved images.



Using with Cursor
Cursor, an AI-powered code editor, can be used to develop, test, and interact with the server:

Open the Project in Cursor:

Launch Cursor and open the imageops-mcp-server project directory.
Cursor will detect the project.toml file and suggest setting up the environment using uv.


Run and Debug:

Use Cursor's terminal to run the server (python image_mcp_server.py).
Leverage Cursor's AI features to debug code, add new tools, or modify existing ones.


Test Tools:

Create a test script in Cursor to call the server tools programmatically:from mcp.client import AsyncClient
import asyncio

async def test_tools():
    async with AsyncClient(transport='stdio') as client:
        result = await client.call_tool('fetch_image', {'image_source': 'https://ultralytics.com/images/bus.jpg'})
        print(result)

asyncio.run(test_tools())


Run the script in Cursor to verify tool functionality.


AI Assistance:

Use Cursor's AI features to generate additional tools, optimize code, or document functions.
For example, ask Cursor to "Add a new tool for image rotation" to extend the server.



Project Structure
imageops-mcp-server/
├── data/                    # Directory for log files
├── Temp/                    # Directory for temporary files
├── output/                  # Directory for processed images
├── image_mcp_server.py      # Main server script
├── project.toml             # Project configuration and dependencies
└── README.md                # Project documentation

Dependencies
The project relies on the following Python packages (specified in project.toml):

asyncio>=3.4.3: For asynchronous programming.
fastmcp>=2.9.0: For the FastMCP server framework.
httpx>=0.28.1: For HTTP requests to fetch images.
opencv-contrib-python>=4.11.0.86: For Canny edge detection and image processing.
pillow>=11.2.1: For image manipulation (cropping, format conversion).
ultralytics>=8.3.162: For YOLOv8 and YOLOv11 models.
mcp[cli]==1.9.4: For the MCP CLI and development tools.
Others: ipykernel, logging, matplotlib, nest-asyncio, mcp-use.

Troubleshooting

Dependency Issues: Ensure uv has installed all dependencies correctly. Run uv pip list to verify.
YOLO Model Errors: If YOLO models fail to load, ensure ultralytics is installed and the model files (yolov8n.pt, yolo11n-seg.pt) are accessible.
File Not Found: Verify that the image source path or URL is correct and accessible.
Permission Errors: Ensure write permissions for the data, Temp, and output directories.
Timeout Errors: For URL-based images, check network connectivity or increase the httpx timeout in the code.

Contributing
Contributions are welcome! To contribute:

Fork the repository.
Create a new branch (git checkout -b feature/your-feature).
Make changes and commit (git commit -m "Add your feature").
Push to the branch (git push origin feature/your-feature).
Open a pull request.

Please ensure code follows PEP 8 standards and includes appropriate tests.
License
This project is licensed under the MIT License. See the LICENSE file for details.
Contact
For questions or support, contact [your-email@example.com] or open an issue on the GitHub repository.