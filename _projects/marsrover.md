---
layout: page
title: Mars Rover
description: 2nd Year EIE Design Project
img: assets/img/marsrover/marsrover.png
importance: 2
category: robotics
---

<a href="https://github.com/alexpondaven/MarsRover">GitHub</a>

This 2nd year EIE design project at Imperial involved creating a Mars Rover. The Mars Rover needed to observe and navigate the environment while also being controlled remotely. It was done in a group of 6, each creating their own subsystem. I worked on vision subsystem, which involved programming and designing the system on a DE10-LITE FPGA to interface with the D8M camera. The task of the subsystem was to track coloured ping pong balls (red, pink, yellow, green, blue) in the rover's view and either follow or avoid them according to the mode controlled by other subsystems.

This project was all done remotely, with team members all over the world and only one person with all the rover equipment to put it all together.

I initially wanted to implement ML algorithms like YOLO to detect the bounding boxes, which may have led to improved accuracy and generalisability to different lighting conditions. I quickly realised the memory limitations of the FPGA would make this very difficult to accomplish, so I stuck with classic computer vision methods implemented in Verilog.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="/assets/img/marsrover/marsrover_cvpipeline.png" title="team pic" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Computer vision pipeline
</div>

The processor handles the image from the camera as a stream of pixel values, so previous scan lines needed to be stored in buffers to implement convolutions. The final output was the detected bounding boxes in the image with its associated colour, which could be sent to the control subsystem to help the logic calculation. The video stream with the detections were also sent to a web console to be displayed live on a website.

In the end, the rover could detect all the different coloured ping pong balls as long as the HSV thresholds were tuned correctly for the lighting conditions. This was made easy through an interface to change these values on the console.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="/assets/img/marsrover/marsrover_console.png" title="team pic" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Mars Rover Web Console
</div>

See <a href="/assets/pdf/marsrover_report.pdf" target="_blank">report</a> for more in-depth analysis.


Demo video:
<iframe width="560" height="315" src="https://www.youtube.com/embed/cwwv1ob5XVA" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>