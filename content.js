// ---------------- URL CHECK ----------------

const url = window.location.href;

fetch("http://127.0.0.1:8000/predict_website", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({ url: url })
})
.then(res => res.json())
.then(data => {

    console.log("Prediction:", data);

    if (data.prediction === 1) {

        // Create overlay instead of deleting page
        const overlay = document.createElement("div");
        overlay.style.position = "fixed";
        overlay.style.top = "0";
        overlay.style.left = "0";
        overlay.style.width = "100%";
        overlay.style.height = "100%";
        overlay.style.background = "black";
        overlay.style.color = "red";
        overlay.style.display = "flex";
        overlay.style.justifyContent = "center";
        overlay.style.alignItems = "center";
        overlay.style.fontSize = "30px";
        overlay.style.zIndex = "999999";

        overlay.innerHTML = "⚠️ PHISHING WEBSITE BLOCKED";

        document.body.appendChild(overlay);
    }

})
.catch(err => console.log("Backend error:", err));


// ---------------- EMAIL DETECTION ----------------

if (window.location.hostname.includes("mail.google.com")) {

    let lastChecked = "";

    setInterval(() => {

        let emailText = document.body.innerText.slice(0, 2000); 
        // limit text (performance improve)

        if (emailText === lastChecked) return; // avoid duplicate check

        lastChecked = emailText;

        fetch("http://127.0.0.1:8000/predict_email", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({ email_text: emailText })
        })
        .then(res => res.json())
        .then(data => {

            if (data.prediction === 1) {

                // Show alert only once
                if (!document.getElementById("phishing-alert")) {

                    const alertBox = document.createElement("div");
                    alertBox.id = "phishing-alert";

                    alertBox.style.position = "fixed";
                    alertBox.style.bottom = "20px";
                    alertBox.style.right = "20px";
                    alertBox.style.background = "red";
                    alertBox.style.color = "white";
                    alertBox.style.padding = "15px";
                    alertBox.style.borderRadius = "10px";
                    alertBox.style.zIndex = "999999";

                    alertBox.innerText = "⚠️ Phishing Email Detected!";

                    document.body.appendChild(alertBox);
                }
            }

        })
        .catch(err => console.log("Email check error:", err));

    }, 5000);
}