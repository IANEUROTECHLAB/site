const cursor = document.querySelector('.cursor');
const animatedIcons = document.querySelector('.animated-icons');
let isMobile = window.innerWidth <= 768;

document.addEventListener('mousemove', e => {
    cursor.style.left = e.clientX + 'px';
    cursor.style.top = e.clientY + 'px';
    cursor.classList.add('show');
});

document.addEventListener('mouseover', e => {
    if (e.target.classList.contains('cta-btn') || e.target.classList.contains('logo') || e.target.closest('.social-icons')) {
        cursor.classList.add('hover');
    }
});

document.addEventListener('mouseout', () => {
    cursor.classList.remove('hover');
});

function createIcon() {
    if (animatedIcons.children.length >= (isMobile ? 5 : 15)) return;

    const icon = document.createElement('div');
    icon.classList.add('animated-icon');

    const size = Math.random() * 30 + 10;
    icon.style.width = `${size}px`;
    icon.style.height = `${size}px`;

    const startX = Math.random() * 100;
    const startY = Math.random() * 100;
    icon.style.left = `${startX}%`;
    icon.style.top = `${startY}%`;

    const endX = Math.random() * 100;
    const endY = Math.random() * 100;
    const duration = Math.random() * 10 + 5;

    icon.animate([
        { transform: `translate(0, 0)` },
        { transform: `translate(${endX - startX}%, ${endY - startY}%)` }
    ], {
        duration: duration * 1000,
        iterations: Infinity,
        direction: 'alternate',
        easing: 'ease-in-out'
    });

    animatedIcons.appendChild(icon);
}

function updateIcons() {
    isMobile = window.innerWidth <= 768;
    while (animatedIcons.children.length > (isMobile ? 5 : 15)) {
        animatedIcons.removeChild(animatedIcons.lastChild);
    }
}

window.addEventListener('resize', updateIcons);

setInterval(createIcon, 1000);

// Countdown Timer
const countDownDate = new Date(2024, 11, 1, 0, 0, 0).getTime();

const countdownFunction = setInterval(() => {
    const now = new Date().getTime();
    const distance = countDownDate - now;

    const days = Math.floor(distance / (1000 * 60 * 60 * 24));
    const hours = Math.floor((distance % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60));
    const minutes = Math.floor((distance % (1000 * 60 * 60)) / (1000 * 60));
    const seconds = Math.floor((distance % (1000 * 60)) / 1000);

    document.getElementById("days").innerHTML = days.toString().padStart(2, '0');
    document.getElementById("hours").innerHTML = hours.toString().padStart(2, '0');
    document.getElementById("minutes").innerHTML = minutes.toString().padStart(2, '0');
    document.getElementById("seconds").innerHTML = seconds.toString().padStart(2, '0');

    if (distance < 0) {
        clearInterval(countdownFunction);
        document.getElementById("days").innerHTML = "00";
        document.getElementById("hours").innerHTML = "00";
        document.getElementById("minutes").innerHTML = "00";
        document.getElementById("seconds").innerHTML = "00";
    }
}, 1000);
