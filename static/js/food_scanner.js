(() => {
  const form = document.querySelector('#food-scan-form');
  const input = document.querySelector('#food-image');
  const drop = document.querySelector('#food-drop-zone');
  const preview = document.querySelector('#food-preview');
  const message = document.querySelector('#food-upload-message');
  const button = form.querySelector('button');
  const resultsPanel = document.querySelector('#food-results');
  const predictionList = document.querySelector('#prediction-list');
  const deviceLabel = document.querySelector('#food-model-device');
  const selectedName = document.querySelector('#selected-food-name');
  const gramsInput = document.querySelector('#serving-grams');
  const note = document.querySelector('#nutrition-note');
  const nutritionForm = document.querySelector('#nutrition-form');
  const lookupButton = document.querySelector('#lookup-nutrition');
  const values = {
    calories: document.querySelector('#calories-value'),
    protein: document.querySelector('#protein-value'),
    carbs: document.querySelector('#carbs-value'),
    fat: document.querySelector('#fat-value')
  };
  let state = null;

  const allowed = ['image/jpeg', 'image/png', 'image/webp'];
  const servingGrams = () => Math.max(1, Number(gramsInput.value || 100));
  const scale = per100 => {
    const factor = servingGrams() / 100;
    return Object.fromEntries(['calories', 'protein', 'carbs', 'fat'].map(key => [key, Number(((per100[key] || 0) * factor).toFixed(1))]));
  };
  const renderNutrition = () => {
    if (!state || !state.nutrition) return;
    state.nutrition.serving_grams = servingGrams();
    state.nutrition.totals = scale(state.nutrition.per_100g);
    for (const key of Object.keys(values)) values[key].textContent = state.nutrition.totals[key];
    note.textContent = state.nutrition.note || 'Nutrition values are estimates.';
  };
  const setNutrition = nutrition => {
    state.nutrition = nutrition;
    renderNutrition();
  };
  const selectedFoodName = () => selectedName.value.trim();
  const showPredictions = predictions => {
    predictionList.innerHTML = '';
    predictions.forEach((item, index) => {
      const row = document.createElement('button');
      row.type = 'button';
      row.className = 'prediction-option';
      row.innerHTML = `<strong>${item.display_name}</strong><span>${item.confidence_percent}%</span>`;
      row.addEventListener('click', () => {
        selectedName.value = item.display_name;
        state.selected_prediction = item;
      });
      predictionList.appendChild(row);
      if (index === 0) {
        selectedName.value = item.display_name;
        state.selected_prediction = item;
      }
    });
  };
  const showFile = () => {
    const file = input.files[0];
    if (!file) {
      preview.hidden = true;
      return;
    }
    if (!allowed.includes(file.type)) {
      message.textContent = 'Choose a JPG, PNG, or WebP image.';
      input.value = '';
      preview.hidden = true;
      return;
    }
    preview.src = URL.createObjectURL(file);
    preview.hidden = false;
    message.textContent = `Selected: ${file.name}`;
  };

  input.addEventListener('change', showFile);
  ['dragenter', 'dragover'].forEach(event => drop.addEventListener(event, e => {
    e.preventDefault();
    drop.classList.add('dragging');
  }));
  ['dragleave', 'drop'].forEach(event => drop.addEventListener(event, e => {
    e.preventDefault();
    drop.classList.remove('dragging');
  }));
  drop.addEventListener('drop', e => {
    input.files = e.dataTransfer.files;
    showFile();
  });

  form.addEventListener('submit', async e => {
    e.preventDefault();
    const file = input.files[0];
    if (!file) return;
    button.disabled = true;
    message.textContent = 'Scanning locally...';
    const data = new FormData();
    data.append('image', file);
    try {
      const response = await fetch('/api/food/scan', {method: 'POST', body: data});
      const result = await response.json();
      if (!response.ok) throw Error(result.error || 'Scan failed.');
      state = result;
      resultsPanel.hidden = false;
      deviceLabel.textContent = `${result.model} · ${result.device}`;
      showPredictions(result.predictions || []);
      if (result.nutrition) setNutrition(result.nutrition);
      if (result.nutrition_error) note.textContent = result.nutrition_error;
      message.textContent = 'Scan complete.';
    } catch (error) {
      message.textContent = error.message;
    } finally {
      button.disabled = false;
    }
  });

  gramsInput.addEventListener('input', renderNutrition);
  const lookupNutrition = async trigger => {
    const foodName = selectedFoodName();
    if (!foodName) {
      note.textContent = 'Scan a food image and select a prediction before looking up nutrition.';
      return;
    }
    trigger.disabled = true;
    note.textContent = 'Looking up nutrition...';
    try {
      const response = await fetch('/api/food/nutrition', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({food_name: foodName, serving_grams: servingGrams()})
      });
      const result = await response.json();
      if (!response.ok) throw Error(result.error || 'Nutrition lookup failed.');
      setNutrition(result.nutrition);
    } catch (error) {
      note.textContent = error.message;
    } finally {
      trigger.disabled = false;
    }
  };

  lookupButton.addEventListener('click', () => lookupNutrition(lookupButton));

  nutritionForm.addEventListener('submit', async e => {
    e.preventDefault();
    if (!state || !state.nutrition) {
      note.textContent = 'Add nutrition values before saving.';
      return;
    }
    renderNutrition();
    try {
      const response = await fetch('/api/food/log', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          dish_name: selectedName.value,
          selected_label: state.selected_prediction ? state.selected_prediction.label : selectedName.value,
          confidence: state.selected_prediction ? state.selected_prediction.confidence : null,
          nutrition: state.nutrition
        })
      });
      const result = await response.json();
      if (!response.ok) throw Error(result.error || 'Food log save failed.');
      note.textContent = result.message;
    } catch (error) {
      note.textContent = error.message;
    }
  });
})();
