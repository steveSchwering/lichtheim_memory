# lichtheim_memory

The following is a repository of the Lichtheim-memory model, an artificial neural network designed to conduct language comprehension, production, and repetition. The model itself is a custom recurrent neural network written in pytorch. A full description of the model, application of the model to language and memory tasks, and examination of the model's responses can be found in [my dissertation](https://www.proquest.com/openview/b7282c9cd4db7a49e2b98540b3086622/1?pq-origsite=gscholar&cbl=18750&diss=y).

A visualization of the Lichtheim-memory model can be found below:
<img width="468" alt="image" src="https://github.com/steveSchwering/lichtheim_memory/assets/30991528/e1e75eba-e615-43b7-9dbc-0f0d4ac8a5e0">

In this model, forward connections are represented by solid arrows, backward connections through time are represented by dashed arrows, and hidden layers by circles. Input and output layers are represented by boxes.

The repository itself is split into two sections. The first section, in the `artificial_language` folder, contains code to generate the artificial language on which the model was trained. The second section, in the `lichtheim-memory` folder, contains code to train and test the neural network.
