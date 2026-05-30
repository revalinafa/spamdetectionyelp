document
  .getElementById("predictionForm")
  .addEventListener("submit", async (e) => {
    e.preventDefault();

    const reviewText = document.getElementById("reviewText").value.trim();
    if (!reviewText) {
      alert("Please enter a review text");
      return;
    }

    const submitButton = e.target.querySelector(".submit-button");
    submitButton.disabled = true;
    submitButton.textContent = "Analyzing...";

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ text: reviewText }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      // Display result
      const resultContainer = document.getElementById("resultContainer");
      const resultBadge = document.getElementById("resultBadge");
      const resultNote = document.getElementById("resultNote");

      const isSpam = data.prediction === "Spam";
      resultBadge.className = "result-badge " + (isSpam ? "spam" : "notspam");
      resultBadge.textContent = data.prediction;
      resultNote.textContent = isSpam
        ? "This review has been classified as spam. 🚨"
        : "This review appears to be legitimate. ✅";

      resultContainer.style.display = "block";
      resultContainer.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (error) {
      console.error("Error:", error);
      alert("Error analyzing review. Please try again.");
    } finally {
      submitButton.disabled = false;
      submitButton.textContent = "Analyze Review";
    }
  });
