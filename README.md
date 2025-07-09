# ImageOps MCP Server

`imageops-mcp-server` is a FastMCP-based Python server designed for advanced image processing tasks. It provides a suite of tools for fetching, manipulating, and analyzing images from URLs or local file paths. The server leverages the YOLOv8 and YOLOv11 models for object detection and segmentation, OpenCV for edge detection, and PIL for image format conversion and cropping. The project is built to be extensible, efficient, and compatible with modern AI-driven workflows, making it suitable for integration with tools like Claude Desktop and Cursor.

## Features

- **Image Fetching**: Retrieve images from URLs or local files, with automatic compression for large images (<1MB).
- **Format Conversion**: Convert images to JPEG, PNG, or GIF formats with optional saving to an output directory.
- **Object Detection**: Perform object detection using YOLOv8, returning labels, confidence scores, and bounding box coordinates.
- **Object Segmentation**: Apply YOLOv11 segmentation to overlay colored masks on detected objects with customizable transparency.
- **Image Cropping**: Crop specific regions of an image using bounding box coordinates, compatible with YOLO detection results.
- **Canny Edge Detection**: Detect edges in images using OpenCV's Canny algorithm with adjustable thresholds.
- **Logging**: Comprehensive logging to both console and file with timestamped logs for debugging and monitoring.
- **Directory Management**: Automatically creates necessary directories (`data`, `Temp`, `output`) for logs and processed images.
- **Integration Ready**: Designed for use with AI tools like Claude Desktop and Cursor for seamless workflow integration.

## Prerequisites

To set up and run the `imageops-mcp-server`, ensure you have the following installed:

- **Python**: Version 3.13 or higher
- **uv**: A modern Python package and project manager (recommended for dependency management)
- **Git**: For cloning the repository
- **Claude Desktop** (optional): For interacting with the server in an AI-driven workflow
- **Cursor** (optional): For code editing and integration with AI assistants
- **Operating System**: Compatible with Windows, macOS, or Linux

## Setup Instructions

Follow these steps to set up the project using `uv`:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/imageops-mcp-server.git
   cd imageops-mcp-server
   ```

2. **Install uv**:
   If you don't have `uv` installed, follow the official installation instructions:
   ```bash
   # On macOS/Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # On Windows (PowerShell)
   iwr -useb https://astral.sh/uv/install.ps1 | iex
   ```

3. **Create a Virtual Environment and Install Dependencies**:
   Use `uv` to set up the project environment and install dependencies specified in `project.toml`:
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   uv pip install .
   ```

   This command reads the `project.toml` file and installs dependencies like `fastmcp`, `httpx`, `ultralytics`, `opencv-contrib-python`, `pillow`, and others.

4. **Verify Directory Structure**:
   The server automatically creates the following directories if they don't exist:
   - `./data`: Stores log files (e.g., `DD-MM-YY.log`).
   - `./Temp`: Temporary storage for intermediate files.
   - `./output`: Stores processed images (e.g., converted, cropped, segmented, or edge-detected images).

   Ensure you have write permissions in the project directory.

## Running the Server

To start the FastMCP server, run the main script:

```bash
uv run image_loader.py
```

The server runs using the `stdio` transport, allowing it to process requests from compatible clients, such as Claude Desktop or custom scripts.

## Testing with `mcp dev`

The `mcp dev` command (part of the `mcp` CLI) allows you to test the server's tools interactively. Follow these steps:

1. **Ensure Dependencies**:
   Confirm that `mcp[cli]` is installed (included in `project.toml`).

2. **Run `mcp dev`**:
   In the project directory, activate the virtual environment and start the development tool:
   ```bash
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   mcp dev image_loader.py
   ```

3. **Test Tools**:
   The `mcp dev` interface allows you to call the server's tools directly. For example:
   - **Fetch Image**:
     ```bash
     fetch_image "https://ultralytics.com/images/bus.jpg"
     ```
     Returns an `Image` object or error message if the fetch fails.
   - **Get Image Dimensions and Save**:
     ```bash
     get_image_dimensions_and_save "https://ultralytics.com/images/bus.jpg"
     ```
     Returns a dictionary with dimensions and saved path.
   - **Convert Image Format**:
     ```bash
     convert_image_format "https://ultralytics.com/images/bus.jpg" "png" true
     ```
     Converts the image to PNG and saves it.
   - **Detect Objects**:
     ```bash
     detect_objects_yolo "https://ultralytics.com/images/bus.jpg" true
     ```
     Performs YOLOv8 object detection and saves the annotated image.
   - **Crop Object**:
     ```bash
     crop_the_object "https://ultralytics.com/images/bus.jpg" 100 150 200 300 true
     ```
     Crops the specified region and saves it.
   - **Segment Objects**:
     ```bash
     segment_objects_yolo "https://ultralytics.com/images/bus.jpg" 0.5 true
     ```
     Performs YOLOv11 segmentation with 0.5 transparency.
   - **Canny Edge Detection**:
     ```bash
     canny_edge_detection "https://ultralytics.com/images/bus.jpg" 100 200 true
     ```
     Applies Canny edge detection with specified thresholds.

4. **Check Outputs**:
   - Logs are saved in the `./data` directory (e.g., `09-07-25.log`).
   - Processed images are saved in the `./output` directory with timestamped filenames (e.g., `image_2025-07-09_171823.jpg`).

## Connecting to Claude Desktop

To connect the `imageops-mcp-server` to Claude Desktop and use its tools:

1. **Start the Server**:
   Ensure the server is running in your project directory:
   ```bash
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   uv run python image_mcp_server.py
   ```

2. **Configure Claude Desktop**:
   - Open Claude Desktop and navigate to **Settings** (typically found in the top navigation bar or menu).
   - Go to **Features** > **MCP Servers**.
   - Click **+ Add New MCP Server** to add a new server configuration.
   - Enter the following configuration, replacing `/path/to/imageops-mcp-server` with the absolute path to your project directory:
     ```json
     {
       "mcpServers": {
         "imageops": {
           "command": "uv",
           "args": ["--directory", "/path/to/imageops-mcp-server", "run", "python", "image_mcp_server.py"]
         }
       }
     }
     ```
   - Save the configuration. Claude Desktop will use `uv` to run the server in the specified directory with the `stdio` transport.

3. **Interact with the Server**:
   - In Claude Desktop's chat interface, you can send requests to the server by invoking its tools using natural language prompts or structured JSON.
   - Claude Desktop will communicate with the server and display the results, such as JSON responses or references to saved images in the `./output` directory.

4. **Sample Prompts**:
   - **Fetch an Image**:
     ```
     Fetch the image from https://ultralytics.com/images/bus.jpg and process it.
     ```
     *Expected Response*: Returns an `Image` object or confirmation of processing, with the image saved in `./output`.
   - **Detect Objects**:
     ```
     Perform object detection on https://ultralytics.com/images/bus.jpg using YOLOv8 and save the result.
     ```
     *Expected Response*: Returns a JSON dictionary with detected objects, bounding boxes, confidence scores, and the saved path of the annotated image.
   - **Convert Image Format**:
     ```
     Convert https://ultralytics.com/images/bus.jpg to PNG format and save it.
     ```
     *Expected Response*: Returns an `Image` object in PNG format and confirms the saved path.
   - **Segment Objects**:
     ```
     Segment objects in https://ultralytics.com/images/bus.jpg with 0.5 transparency and save the output.
     ```
     *Expected Response*: Returns a JSON dictionary with the segmented image, detection details, and saved path.
   - **Crop an Object**:
     ```
     Crop the region (100, 150, 200, 300) from https://ultralytics.com/images/bus.jpg and save it.
     ```
     *Expected Response*: Returns a JSON dictionary with the cropped image and saved path.
   - **Canny Edge Detection**:
     ```
     Apply Canny edge detection to https://ultralytics.com/images/bus.jpg with thresholds 100 and 200, and save the result.
     ```
     *Expected Response*: Returns a JSON dictionary with the edge-detected image and saved path.

5. **View Results**:
   - Claude Desktop will display the JSON response, including detection details, image data, or file paths.
   - Check the `./output` directory for saved images (e.g., `image_2025-07-09_174523.jpg`).

## Connecting to Cursor

To connect the `imageops-mcp-server` to Cursor and use its tools:

1. **Open the Project in Cursor**:
   - Launch Cursor and open the `imageops-mcp-server` project directory.
   - Cursor will detect the `project.toml` file and prompt you to set up the environment using `uv`. Follow the prompt or manually run:
     ```bash
     uv venv
     source .venv/bin/activate  # On Windows: .venv\Scripts\activate
     uv pip install .
     ```

2. **Configure Cursor**:
   - Open Cursor and go to **Settings** (Navbar > **Cursor Settings**).
   - Navigate to **Features** >.mob
   - Click **+ Add New MCP Server**.
   - Add the following configuration, replacing `/path/to/imageops-mcp-server` with the absolute path to your project directory:
     ```json
     {
       "mcpServers": {
         "imageops": {
           "command": "uv",
           "args": ["--directory", "/path/to/imageops-mcp-server", "run", "python", "image_mcp_server.py"]
         }
       }
     }
     ```
   - Save the configuration. This allows Cursor to communicate with the server via `stdio`.

3. **Interact with the Server**:
   - Use Cursor's chat interface or a test script to send requests to the server.
   - Alternatively, create a Python script in Cursor to call tools programmatically:
     ```python
     from mcp.client import AsyncClient
     import asyncio

     async def test_tools():
         async with AsyncClient(transport='stdio') as client:
             result = await client.call_tool('detect_objects_yolo', {
                 'image_source': 'https://ultralytics.com/images/bus.jpg',
                 'save': True
             })
             client.write(result)

     asyncio.run(test_tools())
     ```
   - Run the script in Cursor's terminal to test server functionality.

4. **Sample Prompts**:
   - **Fetch an Image**:
     ```
     Please fetch and process the image from https://ultralytics.com/images/bus.jpg.
     ```
     *Expected Response*: Cursor's AI will trigger the `fetch_image` tool and return the processed image data or a confirmation.
   - **Object Detection**:
     ```
     Detect objects in https://ultralytics.com/images/bus.jpg using YOLOv8 and save the annotated image.
     ```
     *Expected Response*: Returns a JSON response with detection details and the path to the saved annotated image.
   - **Image Conversion**:
     ```
     Convert https://ultralytics.com/images/bus.jpg to GIF format and save it.
     ```
     *Expected Response*: Confirms conversion to GIF and provides the saved path.
   - **Object Segmentation**:
     ```
     Segment objects in https://ultralytics.com/images/bus.jpg with 0.7 transparency and save the result.
     ```
     *Expected Response*: Returns a JSON dictionary with the segmented image, detections, and saved path.
   - **Crop an Object**:
     ```
     Crop the region (x_min: 100, y_min: 150, x_max: 200, y_max: 300) from https://ultralytics.com/images/bus.jpg and save it.
     ```
     *Expected Response*: Returns a JSON dictionary with the cropped image and saved path.
   - **Canny Edge Detection**:
     ```
     Perform Canny edge detection on https://ultralytics.com/images/bus.jpg with thresholds 50 and 150, and save the output.
     ```
     *Expected Response*: Returns a JSON dictionary with the edge-detected image and saved path.

5. **AI Assistance**:
   - Use Cursor's AI chat to generate new tools or optimize existing code. For example, prompt:
     ```
     Add a new MCP tool to rotate images by a specified angle.
     ```
     Cursor's AI can generate the code, which you can add to `image_mcp_server.py`.

6. **View Results**:
   - Cursor will display the JSON response or script output in the terminal or chat interface.
   - Check the `./output` directory for saved images.


## Project Structure

```plaintext
imageops-mcp-server/
├── data/                    # Directory for log files
├── Temp/                    # Directory for temporary files
├── output/                  # Directory for processed images
├── image_mcp_server.py      # Main server script
├── project.toml             # Project configuration and dependencies
└── README.md                # Project documentation
```

## Dependencies

The project relies on the following Python packages (specified in `project.toml`):

- `asyncio>=3.4.3`: For asynchronous programming.
- `fastmcp>=2.9.0`: For the FastMCP server framework.
- `httpx>=0.28.1`: For HTTP requests to fetch images.
- `opencv-contrib-python>=4.11.0.86`: For Canny edge detection and image processing.
- `pillow>=11.2.1`: For image manipulation (cropping, format conversion).
- `ultralytics>=8.3.162`: For YOLOv8 and YOLOv11 models.
- `mcp[cli]==1.9.4`: For the MCP CLI and development tools.
- Others: `ipykernel`, `logging`, `matplotlib`, `nest-asyncio`, `mcp-use`.

## Troubleshooting

- **Dependency Issues**: Ensure `uv` has installed all dependencies correctly. Run `uv pip list` to verify.
- **YOLO Model Errors**: If YOLO models fail to load, ensure `ultralytics` is installed and the model files (`yolov8n.pt`, `yolo11n-seg.pt`) are accessible.
- **File Not Found**: Verify that the image source path or URL is correct and accessible.
- **Permission Errors**: Ensure write permissions for the `data`, `Temp`, and `output` directories.
- **Timeout Errors**: For URL-based images, check network connectivity or increase the `httpx` timeout in the code.

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Make changes and commit (`git commit -m "Add your feature"`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

Please ensure code follows PEP 8 standards and includes appropriate tests.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

