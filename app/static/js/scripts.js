document.getElementById('categorie').addEventListener('change', function() {
    const selectedKPI = this.value;
    const selectedOption = document.getElementById('experience').value;
    visualizeJob(selectedKPI, selectedOption);
});

document.getElementById('experience').addEventListener('change', function() {
    const selectedKPI = document.getElementById('categorie').value;
    const selectedOption = this.value;
    
    visualizeJob(selectedKPI, selectedOption);
    
});

document.getElementById('no-button').addEventListener('click', function() {
    getResponse("no");
});

document.getElementById('yes-button').addEventListener('click', function() {
    getResponse("yes");
});



function getResponse(button){
    const formData = new FormData();
    formData.append('response', button);
    fetch('/response', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
      if(data.job_content){
        const jobContent = document.getElementById("job-content")
        jobContent.innerHTML = data.job_content;
      }
      else{

        let idx = 0
        const jobContent = document.getElementById("job-content")
        const cvContent = document.getElementById("cv-content")
        jobContent.innerHTML = data.requirements;

        const yesNoButtons = document.getElementById("yes-no-buttons");
        yesNoButtons.style.display = "None"; // Show the buttons when job content is updated
        let cvDict = {}

        const length = Object.keys(data.cvs).length;
        const stars_div = document.getElementById("stars");
        
        const matchButtons = document.getElementById("match-buttons");
        console.log(length)
        function processCV(idx) {
            if (idx >= length) return; // Exit condition
        
            cvContent.innerHTML = data.cvs[idx];
            matchButtons.style.display = "flex";
            stars_div.style.display = "flex";


            function handleMatchClick(match, stars) {
                console.log(stars)
                cvDict[idx]["match"] = match === 'yes'
                matchButtons.style.display = "none"; // Hide buttons to prevent multiple clicks
            };

            function handleStarsChange(stars) {
                cvDict[idx]["stars"] = stars;
                stars_div.style.display = "none"; 
                processCV(idx+1);
            };
            
            document.getElementById('match-no-button').onclick = function() {
                match = 'no';
                handleMatchClick(match);
            };
        
            document.getElementById('match-yes-button').onclick = function() {
                match = "yes";
                handleMatchClick(match);
            };
            document.getElementById('stars_selection').addEventListener('change', handleStarsChange(this.value));

        }
        processCV(0);
        console.log(cvDict)
      }
    });
}


function visualizeJob(kpi, option) {
    const formData = new FormData();
    formData.append('categorie', kpi);
    formData.append('experience', option);
    fetch('/visualize', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
       const jobContent = document.getElementById("job-content")
       jobContent.innerHTML = data.job_content;

       const yesNoButtons = document.getElementById("yes-no-buttons");
       console.log(yesNoButtons)
       yesNoButtons.style.display = "flex"; // Show the buttons when job content is updated


    });
}
