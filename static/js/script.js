document.getElementById('load-more-button').addEventListener('click', function() {
    fetch('/get_more_cards', {
        method: 'POST',
        headers: {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        },
    })
    .then(response => response.json())
    .then(data => {
        data.cards.forEach(card => {
            let newCard = document.createElement('div');
            newCard.className = 'card';

            let image = document.createElement('img');
            image.src = card.image_url;
            image.alt = `Карточка ${card.id}`;

            let cardText = document.createElement('div');
            cardText.className = 'card-text';
            cardText.textContent = card.title;

            newCard.appendChild(image);
            newCard.appendChild(cardText);
            document.getElementById('cards-container').appendChild(newCard);
        });
    });
});