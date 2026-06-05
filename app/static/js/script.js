const form = document.getElementById("predictionForm");

form.addEventListener("submit", async (e) => {

    e.preventDefault();

    const button = document.getElementById("predictBtn");

    button.innerHTML = "Predicting...";
    button.disabled = true;

    const formData = new FormData(form);

    try {

        const response = await fetch("/predict", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        const resultCard =
            document.getElementById("resultCard");

        document.getElementById(
            "predictionValue"
        ).innerText =
            Number(data.prediction).toFixed(2);

        resultCard.classList.remove("hidden");

    } catch(error){

        alert("Prediction Failed");

        console.error(error);
    }

    button.innerHTML = "Predict Performance";
    button.disabled = false;
});