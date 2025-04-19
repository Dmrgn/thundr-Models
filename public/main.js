// public/main.js

document.addEventListener("DOMContentLoaded", () => {
    const sessionForm = document.getElementById("session-form");
    const sourceDirInput = document.getElementById("sourceDir");
    const destDirInput = document.getElementById("destDir");
    const startSessionBtn = document.getElementById("start-session-btn");

    const sortingSection = document.getElementById("sorting-section");
    const imageDisplay = document.getElementById("image-display");
    const targetFolderContainer = document.getElementById("target-folder");
    const moveBtn = document.getElementById("move-btn");
    const skipBtn = document.getElementById("skip-btn");
    const progressDisplay = document.getElementById("progress");

    let images = [];
    let currentIndex = 0;
    let sourceDir = "";
    let destDir = "";
    let destDirs = [];

    async function loadImages() {
        try {
            const response = await fetch(`/images?sourceDir=${encodeURIComponent(sourceDir)}&destDir=${encodeURIComponent(destDir)}`);
            const data = await response.json();
            images = data.images;
            destDirs = data.destDirs;
            console.log(data);
            // attach dest buttons
            for (const dir of destDirs) {
                const button = document.createElement("button");
                button.addEventListener('click', ()=>{
                    moveCurrentImage(dir);
                })
                button.innerHTML = dir;
                button.classList.add('bg-blue-500', 'text-white', 'px-4', 'py-2', 'rounded', 'hover:bg-blue-600', 'mr-4');
                targetFolderContainer.appendChild(button);
            }
            currentIndex = 0;
            if (images.length === 0) {
                alert("No images found in the provided source directory.");
                return;
            }
            sessionForm.classList.add("hidden");
            sortingSection.classList.remove("hidden");
            loadCurrentImage();
        } catch (error) {
            alert("Error loading images " + error.message);
        }
    }

    function loadCurrentImage() {
        if (currentIndex >= images.length) {
            progressDisplay.textContent = "All images processed.";
            imageDisplay.src = "";
            moveBtn.disabled = true;
            skipBtn.disabled = true;
            return;
        }
        const currentImage = images[currentIndex];
        imageDisplay.src = `/image?sourceDir=${encodeURIComponent(sourceDir)}&filename=${encodeURIComponent(currentImage)}`;
        progressDisplay.textContent = `Image ${currentIndex + 1} of ${images.length}`;
    }

    async function moveCurrentImage(dir) {
        const currentImage = images[currentIndex];
        try {
            const response = await fetch("/move-image", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    sourceDir,
                    destinationDir: destDir,
                    filename: currentImage,
                    targetFolder: dir
                })
            });
            const result = await response.json();
            if (result.success) {
                currentIndex++;
                loadCurrentImage();
            } else {
                alert("Error moving image " + result.error);
            }
        } catch (error) {
            alert("Error " + error.message);
        }
    }

    startSessionBtn.addEventListener("click", () => {
        sourceDir = sourceDirInput.value.trim();
        destDir = destDirInput.value.trim();
        if (!sourceDir || !destDir) {
            alert("Please provide both source and destination directories.");
            return;
        }
        loadImages();
    });

    skipBtn.addEventListener("click", () => {
        currentIndex++;
        loadCurrentImage();
    });
});
