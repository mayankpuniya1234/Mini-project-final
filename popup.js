document.addEventListener("DOMContentLoaded", () => {

    const btn = document.getElementById("checkBtn");
    const urlBox = document.getElementById("urlBox");
    const loader = document.getElementById("loader");
    const resultBox = document.getElementById("resultBox");

    btn.addEventListener("click", () => {

        loader.classList.remove("hidden");
        resultBox.classList.add("hidden");

        chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {

            let url = tabs[0].url;
            urlBox.innerText = url;

            fetch("https://phishing-detection-system-2-rtf8.onrender.com", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({url: url})
            })
            .then(res => res.json())
            .then(data => {

                loader.classList.add("hidden");
                resultBox.classList.remove("hidden");

                const status = document.getElementById("status");
                const fill = document.getElementById("fill");

                if(data.prediction_label === "Phishing"){
                    status.innerText = "⚠️ Phishing Detected";
                    status.className = "status phishing";
                    fill.style.background = "red";
                } else {
                    status.innerText = "✅ Safe Website";
                    status.className = "status safe";
                    fill.style.background = "green";
                }

                document.getElementById("confidenceText").innerText = data.confidence + "%";
                fill.style.width = data.confidence + "%";

                document.getElementById("reason").innerText = data.reason || "-";
                document.getElementById("theory").innerText = data.theory || "-";

            })
            .catch(err => {
                loader.classList.add("hidden");
                alert("Backend error");
                console.error(err);
            });

        });

    });

});