// function onSearch(event){
//     if (event) event.preventDefault(); 

//     const ele = document.getElementById("maintitle");
//     if (ele) {
//         ele.className = "secondary-title";
//     }
// }

document.addEventListener("DOMContentLoaded", () => {
    const title = document.getElementById("maintitle");

    if (title) {
        title.classList.add("secondary-title");
    }
});
