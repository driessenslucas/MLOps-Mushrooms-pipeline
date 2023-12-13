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
const imageElement = document.getElementById('pokemonImage');
const imageDataURL = imageElement.src;

classifyButton.addEventListener('click', function () {
	//get image data from canvas
	const imageDataURL = document.getElementById('pokemonImage').src;
	//send image data to api
	fetch(imageDataURL)
		.then((res) => res.blob())
		.then((blob) => {
			// Create a FormData object
			const formData = new FormData();
			formData.append('img', blob, 'image.png');

			// Send the image file to the FastAPI server
			fetch('http://localhost:8700/upload/image', {
				method: 'POST',
				body: formData,
			})
				.then((response) => response.json())
				.then((data) => {
					console.log(data);
				})
				.catch((error) => {
					console.error('Error:', error);
				});
		});
});
