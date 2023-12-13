// Get a reference to the file input element
const imageUpload = document.getElementById('imageUpload');

// Add an event listener to the file input element
imageUpload.addEventListener('change', function (event) {
	const selectedFile = event.target.files[0]; // Get the selected file

	if (selectedFile) {
		// Create a FileReader
		const reader = new FileReader();

		// Define a function to be executed when the file is loaded
		reader.onload = function (e) {
			const imageDataURL = e.target.result; // Get the data URL of the image
			const pokemonImage = document.getElementById('pokemonImage');

			// Set the src attribute of the img element to display the selected image
			pokemonImage.src = imageDataURL;
		};

		// Read the selected file as a data URL
		reader.readAsDataURL(selectedFile);
	}
});

const classifyButton = document.getElementById('classifyButton');

classifyButton.addEventListener('click', function () {
	//get image data from canvas
	const imageDataURL = document.getElementById('pokemonImage').src;
	//send image data to api
	fetch('https://pokemon-classifier-api.herokuapp.com/classify', {
		method: 'POST',
		headers: {
			'Content-Type': 'application/json',
		},
		body: JSON.stringify({
			image: imageDataURL,
		}),
	});
});
