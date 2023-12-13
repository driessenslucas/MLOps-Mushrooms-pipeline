// Get a reference to the file input element
const imageUpload = document.getElementById('imageUpload');
const classifyButton = document.getElementById('classifyButton');
const soundButton = document.getElementById('soundButton');
const imageElement = document.getElementById('mushroomImage');
const imageDataURL = imageElement.src;

const mushroomName = document.getElementById('mushroomName');
const mushroomType = document.getElementById('mushroomType');
const mushroomDescription = document.getElementById('mushroomDescription');

const mushroomDescriptions = [
	{
		name: 'Agaricus',
		description:
			"This genus is known for mushrooms with a fleshy cap and a number of radiating gills underneath, where spores are produced. They are characterized by chocolate-brown spores and a stem or stipe that elevates the mushroom above its growing substrate. Agaricus includes widely consumed mushrooms like the common 'button' mushroom (Agaricus bisporus) and the field mushroom (A. campestris). However, it's important to be cautious, as some Agaricus species are poisonous, notably those around the yellow-staining mushroom, A. xanthodermus, and the deadly poisonous A. aurantioviolaceus from Africa. The genus also has a history of being confused with deadly species of Amanita, making it essential to correctly identify mushrooms before consumption.",
	},
	{
		name: 'Amanita',
		description:
			"Unfortunately, I couldn't find specific descriptions for Amanita and the other mushroom families you mentioned. However, it's important to note that many Amanita mushrooms are highly toxic and should be handled with extreme caution. The genus includes some of the most poisonous mushrooms known, such as Amanita phalloides, commonly known as the death cap.",
	},
	{
		name: 'Boletus',
		description:
			'Boletus is another large genus of mushrooms, distinct for their sponge-like layer of tubes on the underside of the cap, instead of gills. Some species in this genus, like Boletus edulis (the porcini mushroom), are highly prized for culinary use. However, like many other mushroom families, Boletus also includes toxic species.',
	},
	{
		name: 'Cortinarius',
		description:
			'This is the largest genus of mushrooms, noted for a cobweb-like partial veil (cortina) when young. Many species within this genus are difficult to identify and some are dangerously toxic.',
	},
	{
		name: 'Entoloma',
		description:
			'Mushrooms in this genus are characterized by their pink spore print, which is unique among the gilled mushrooms. While some species are edible, others can be poisonous.',
	},
	{
		name: 'Hygrocybe',
		description:
			'Known for their bright, waxy caps, mushrooms in this genus are often vividly colored. They are typically found in grasslands and are generally considered non-toxic, but not all are edible.',
	},
	{
		name: 'Lactarius',
		description:
			'These mushrooms are known for producing a milky fluid when cut. The edibility of Lactarius species varies, with some being edible and others being poisonous or unpalatable.',
	},
	{
		name: 'Russula',
		description:
			'This genus is characterized by brittle gills due to the cells breaking apart easily. While some Russula mushrooms are edible, others are not, and their identification can be challenging.',
	},
	{
		name: 'Suillus',
		description:
			'Suillus mushrooms are typically associated with conifer trees and have a slimy cap surface. Many are edible, but as with other mushroom families, accurate identification is crucial.',
	},
];

// Add an event listener to the file input element
imageUpload.addEventListener('change', function (event) {
	const selectedFile = event.target.files[0]; // Get the selected file

	if (selectedFile) {
		// Create a FileReader
		const reader = new FileReader();

		// Define a function to be executed when the file is loaded
		reader.onload = function (e) {
			const imageDataURL = e.target.result; // Get the data URL of the image
			const mushroomImage = document.getElementById('mushroomImage');

			// Set the src attribute of the img element to display the selected image
			mushroomImage.src = imageDataURL;
			classifyButton.click();
		};

		// Read the selected file as a data URL
		reader.readAsDataURL(selectedFile);
	}
});

//possible use for text to speech if you want to add it this way

// const speakDescription = (text) => {
// 	if ('speechSynthesis' in window) {
// 		var msg = new SpeechSynthesisUtterance(text);
// 		window.speechSynthesis.speak(msg);
// 	} else {
// 		// Handle browsers that don't support Web Speech API
// 		alert('Sorry, your browser does not support text to speech!');
// 	}
// };

classifyButton.addEventListener('click', function () {
	//get image data from canvas
	const imageDataURL = document.getElementById('mushroomImage').src;
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
				.then((mushroomFamily) => {
					console.log(mushroomFamily);

					// Update the UI with the received data
					mushroomName.textContent = mushroomFamily || 'Unknown';
					mushroomType.textContent = 'Mushroom Family';
					mushroomDescription.textContent =
						mushroomDescriptions.find((x) => x.name === mushroomFamily)
							.description || 'No description available';

					//make sound button available
					soundButton.style.display = 'block';
				})
				.catch((error) => {
					console.error('Error:', error);
				});
		});
});

let audio = null;

soundButton.addEventListener('click', function () {
	const mushroomName = document.getElementById('mushroomName').textContent;

	if (!audio) {
		audio = new Audio(`./voice-overs/${mushroomName}.mp3`);
		audio.play();
		audio.addEventListener('ended', () => {
			audio = null;
		});
	} else {
		audio.pause();
		audio = null;
	}
});
