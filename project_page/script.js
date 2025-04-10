// Script file for the representation similarity explorer


// initialize global variables
const data_base_url = 'https://sd-similarities.jloos.de/';
const AVAILABLE_STEPS = [0, 20, 50, 100, 200, 500, 999];
let last_tile = null;  // remember the last tile to avoid redundant updates
let last_clicked = { concept_id: null, col: 0, row: 0 };
const available_models = [];
let current_model = null;
let available_positions = {};
let current_position = null;
const repr_cache = {};
const available_concepts = {};
const concepts = [];
let render_counter = 0;  // keep track of canvas updates, so no outdated results are displayed


// setup webworker for similarity computations
const worker = new Worker('worker/webworker.js', { type: 'module' });
worker.jobs = {};
worker.job_counter = 0;
worker.last_calc_promise = new Promise((resolve, reject) => {resolve()});
worker.onmessage = (e) => {
  const { id, data } = e.data;
  const job = worker.jobs[id];
  if (job) {
    // promises should always be resolved and not rejected, so that the calculation promise chain keeps running
    job.resolve(data);
    delete worker.jobs[id];
  } else {
    console.warn('Received message from webworker for unknown job:', e.data);
  }
};
worker.onerror = (e) => {
  console.error('Error in worker:', e);
};


// create a job that can be sent to the webworker
function createWorkerJob(task) {
  const id = worker.job_counter++;
  const job = {id, task};
  const promise = new Promise((resolve, reject) => {
    job.resolve = resolve;
    job.reject = reject;
  });
  // return a function that can be used to send data to the worker
  return {send: (data) => {
    worker.jobs[id] = job;
    worker.postMessage({ id, task, data });
    return promise;
  }};
}


// calculate the similarities between two representations using the webworker
function calcSimilarities(data) {
  const job = createWorkerJob('calc_similarities');
  // wait for the previous calculation job to finish (only one at a time), i.e. build a chain of calculation promises
  const render_counter_backup = render_counter;
  return worker.last_calc_promise = worker.last_calc_promise.then(() => {
    if (render_counter_backup !== render_counter) {
      // if the calculation job is outdated, don't do it
      return {status: 'error', msg: 'outdated'};
    }
    return job.send(data);
  });
}


// fetch a representation from the server and cache it in the webworker
function fetchRepr(data) {
  return createWorkerJob('fetch_repr').send(data);
}


// Helper function to create HTML elements
function createElem(tag, attrs, parent) {
  const elem = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    elem[key] = value;
  }
  parent.appendChild(elem);
  return elem;
}


// setup a concept with its canvas elements and event listeners
function setupConcept(name) {
  if (!available_concepts[name]) return console.error(`Concept ${name} not found`);
  clearPresetSelection();
  const concept = { id: Math.random().toString(36).slice(2, 9), name, step_index: 2, available_steps: [] };

  // create html elements
  const canvas_container = document.getElementById('canvas-container');
  const canvas_row = createElem('div', { className: 'concept-row' }, canvas_container);
  const image_canvas_div = createElem('div', {}, canvas_row);
  const canvas_size = image_canvas_div.clientWidth * 2;
  // const title_p = createElem('p', {}, image_canvas_div);
  // createElem('span', { textContent: `noise step:` }, title_p);
  // const slider = createElem('input', { type: 'range', min: 0, max: AVAILABLE_STEPS.length-1, value: concept.step_index }, title_p);
  // const slider_value = createElem('span', { textContent: '?' }, title_p);
  concept.image_canvas = createElem('canvas', { width: canvas_size, height: canvas_size }, image_canvas_div);
  const tile_canvas_div = createElem('div', {}, canvas_row);
  // const text_p = createElem('p', {}, tile_canvas_div);
  // createElem('span', { textContent: `Similarities, avg: ` }, text_p);
  // concept.text = createElem('span', { textContent: `?` }, text_p);
  concept.tile_canvas = createElem('canvas', { width: canvas_size, height: canvas_size }, tile_canvas_div);
  concept.delete = () => {
    concepts.splice(concepts.findIndex(c => c.id === concept.id), 1);
    canvas_row.remove();
    updateCanvasesWithLastClicked();
    clearPresetSelection();
  }
  createElem('button', { 
    textContent: '×', 
    className: 'delete-concept',
    onclick: concept.delete
  }, canvas_row);

  // if this is the first concept, set last_clicked to it
  if (!last_clicked.concept_id) last_clicked.concept_id = concept.id;

  // get canvas contexts
  concept.image_ctx = concept.image_canvas.getContext('2d');
  concept.tile_ctx = concept.tile_canvas.getContext('2d');

  // get the url of the representation. `null` if the representation is not available yet.
  concept.getUrl = () => concept.available_steps.includes(AVAILABLE_STEPS[concept.step_index]) ? `${data_base_url}representations/${concept.name}/${current_model.short}/${AVAILABLE_STEPS[concept.step_index]}/${current_position}.bin` : null;

  // load the representation from the server
  const getRepr = () => {
    const url = concept.getUrl();
    if (!url) return;
    const { n, m } = current_model.getShapes();
    fetchRepr({ url, n, m })
      .then((data) => {
        if (data.status === 'success') {
          updateCanvasesWithLastClicked();
        } else {
          console.warn('Error while fetching representation:', data.msg);
        }
      })
      .catch(error => {
        console.error(`Error while fetching from ${url}:`, error);
      });
  };

  // update concept object to account for changes in model, step, etc.
  concept.update = () => {
    const concept_info = available_concepts[concept.name];

    // update available steps
    concept.available_steps = Object.keys(concept_info[current_model.short] ?? {}).map(Number).filter(step => AVAILABLE_STEPS.includes(step)).sort((a, b) => a - b);

    // generate missing representations
    if (!concept.getUrl()) {
      generateRepresentations(concept.name, AVAILABLE_STEPS[concept.step_index])
        .then(concept.update)
        .catch(error => console.warn('Error while generating representations: ', error));
    }

    // update slider
    // slider.value = concept.step_index;
    // slider_value.textContent = AVAILABLE_STEPS[concept.step_index];

    // update image und representation
    concept.img.src = '';
    concept.img.src = `${data_base_url}images_small/${concept.name}.jpg`;
    updateCanvasesWithLastClicked();
    getRepr();
  }

  // load image
  concept.img = new Image();
  concept.img.src = `${data_base_url}images_small/${concept.name}.jpg`;
  concept.img.onload = function() {
    concept.image_ctx.globalCompositeOperation = 'destination-over';
    concept.image_ctx.drawImage(concept.img, 0, 0, concept.image_ctx.canvas.width, concept.image_ctx.canvas.height);
    concept.image_ctx.globalCompositeOperation = 'source-over';
    updateCanvasesWithLastClicked();
  };

  // setup event listeners  
  [concept.image_canvas, concept.tile_canvas].forEach(canvas => {
    const moveHandler = function(event) {
      const { col, row } = getMousePos(canvas, event);
      updateCanvases(concept, col, row);
    };
    const clickHandler = function(event) {
      const { col, row } = getMousePos(canvas, event);
      last_clicked.concept_id = concept.id;
      last_clicked.col = col;
      last_clicked.row = row;
      updateCanvasesWithLastClicked();
    };
    canvas.addEventListener('click', clickHandler);
    canvas.addEventListener('mousemove', moveHandler);
    canvas.addEventListener('mouseleave', updateCanvasesWithLastClicked);
    canvas.addEventListener('touchstart', clickHandler);
    canvas.addEventListener('touchmove', clickHandler);
  });
  
  // slider event listener
  // slider.addEventListener('input', event => {
  //   const selectedStep = parseInt(event.target.value);
  //   concept.step_index = selectedStep;
  //   concept.update();
  // });

  // finally, update the concept
  concept.update();

  // concept is ready
  return concept;
}


// draw the grid
function drawGrid(ctx) {
  const { n } = current_model.getShapes();
  const tile_size = ctx.canvas.width / n;
  ctx.strokeStyle = 'black';
  ctx.lineWidth = 2;
  ctx.beginPath();

  // Draw vertical lines
  for (let i = 1; i < n; i++) {
    ctx.moveTo(i * tile_size, 0);
    ctx.lineTo(i * tile_size, ctx.canvas.height);
  }

  // Draw horizontal lines
  for (let i = 1; i < n; i++) {
    ctx.moveTo(0, i * tile_size);
    ctx.lineTo(ctx.canvas.width, i * tile_size);
  }

  ctx.stroke();
}


// draw the similarity matrix
function drawSimilarities({ tile_ctx, text }, similarities) {
  const { n, m } = current_model.getShapes();
  const tile_size = tile_ctx.canvas.width / n;
  const { width, height } = tile_ctx.canvas;
  tile_ctx.clearRect(0, 0, width, height);

  // fill canvas with black background
  // tile_ctx.fillStyle = 'black';
  // tile_ctx.fillRect(0, 0, width, height);

  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const similarity = similarities[i + j * n];
      // draw orange for positive similarity and blue for negative similarity
      tile_ctx.fillStyle = similarity > 0 ? `rgba(255, 165, 0, ${similarity})` : `rgba(0, 165, 255, ${-similarity})`;
      tile_ctx.fillRect(i * tile_size, j * tile_size, tile_size, tile_size);
    }
  }
  // calculate and update average similarity
  // const averageSimilarity = similarities.reduce((a, b) => a + b, 0) / similarities.length;
  // const stdDev = Math.sqrt(similarities.reduce((a, b) => a + (b - averageSimilarity) ** 2, 0) / similarities.length);
  // text.textContent = `${averageSimilarity.toFixed(3)}±${stdDev.toFixed(3)}`;
}


// draw an error message on the canvas
function drawError(concept, msg) {
  const tile_ctx = concept.tile_ctx;
  const { width, height } = tile_ctx.canvas;
  tile_ctx.clearRect(0, 0, width, height);
  tile_ctx.fillStyle = 'black';
  tile_ctx.font = '30px Arial';
  tile_ctx.textAlign = 'center';
  tile_ctx.textBaseline = 'middle';
  const loading_dots_count = Math.floor((Date.now() % 1000) / 250);
  const loading_dots = '.'.repeat(loading_dots_count) + ' '.repeat(3 - loading_dots_count);
  msg = msg.replace('...', loading_dots);
  const lines = msg.split('\n');
  for (let i = 0; i < lines.length; i++) {
    tile_ctx.fillText(lines[i], width / 2, height / 2 - (lines.length-1) * 20 + i * 40);
  }
  // concept.text.textContent = `?`;
}


// Update the canvas highlightings based on the mouse position
function updateCanvases(base_concept, col, row) {
  const { n } = current_model.getShapes();
  if (!base_concept) return;

  // Check if the mouse is inside the canvas
  if (col < 0 || col >= n || row < 0 || row >= n) return;

  // Check if the tile has changed
  const curr_tile = `${base_concept.id}-${col}-${row}`;
  if (curr_tile === last_tile) return;
  last_tile = curr_tile;
  render_counter++;

  // Update the canvases
  const func = document.getElementById('similarity-measure').value;
  const concepts_sorted = concepts.slice().sort((a, b) => b == base_concept);  // sort concepts so that the base concept is drawn first
  for (const concept of concepts_sorted) {
    const tile_size = concept.image_ctx.canvas.width / n;
    // update image canvas
    const img_ctx = concept.image_ctx;
    img_ctx.clearRect(0, 0, img_ctx.canvas.width, img_ctx.canvas.height);
    try {
      img_ctx.drawImage(concept.img, 0, 0, img_ctx.canvas.width, img_ctx.canvas.height);      
    } catch (error) {
      console.warn('Error while drawing image: ', error.toString());
    }
    drawGrid(img_ctx);
    // highlight the currently selected tile
    if (concept === base_concept) {
      img_ctx.strokeStyle = 'rgba(255, 165, 0, 1)';
      img_ctx.lineWidth = 2;
      img_ctx.strokeRect(col * tile_size, row * tile_size, tile_size, tile_size);
      img_ctx.fillStyle = 'rgba(255, 165, 0, 0.7)';
      img_ctx.fillRect(col * tile_size, row * tile_size, tile_size, tile_size);
    }
    // update tile canvas
    const url1 = base_concept.getUrl();
    const url2 = concept.getUrl();
    if (!url1 || !url2) {
      drawError(concept, 'Generating representations...');
      updateCanvasesSoon(base_concept, col, row);  // schedule update to check if representations are loaded and animate loading text
      continue;
    }
    calcSimilarities({ func, repr1_str: url1, repr2_str: url2, row, col })
      .then(data => {
        if (data.status === 'success') {
          drawSimilarities(concept, data.similarities);
          // highlight the most similar tile
          if (concept !== base_concept) {
            const max_sim = Math.max(...data.similarities);
            const max_idx = data.similarities.indexOf(max_sim);
            const max_col = max_idx % n;
            const max_row = Math.floor(max_idx / n);
            img_ctx.fillStyle = 'rgba(165, 255, 0, 0.7)';
            img_ctx.fillRect(max_col * tile_size, max_row * tile_size, tile_size, tile_size);
          }
        } else {
          if (data.msg === 'loading') {
            drawError(concept, 'Loading...');
            updateCanvasesSoon(base_concept, col, row);  // schedule update to check if representations are loaded and animate loading text
            return;
          } else if (data.msg === 'outdated') {
            // just ignore outdated responses
          } else {
            console.warn('Error in webworker while calculating similarities: ', data.msg);
            drawError(concept, 'Error calculating similarities...');
            updateCanvasesSoon(base_concept, col, row);
          }
        }
      })
      .catch((error) => {
        console.warn('Unknown error while calculating similarities: ', error);
        drawError(concept, 'Error calculating similarities...');
        updateCanvasesSoon(base_concept, col, row);
      });
  }
}


// force update of all canvases with the last clicked tile as the base
function updateCanvasesWithLastClicked() {
  last_tile = null;  // force update
  updateCanvases(concepts.find(c => c.id === last_clicked.concept_id), last_clicked.col, last_clicked.row);
}


// Schedule a delayed canvas update
function updateCanvasesSoon(concept, col, row) {
  // force update
  last_tile = null;
  // update now if the last update was more than 100ms ago, to ensure updates even when this function is called often
  if (self.timer && self.last_updated && self.last_updated + 200 < Date.now()) {
    self.last_updated = Date.now();
    updateCanvases(concept, col, row);
  }
  // schedule update in 200ms
  clearTimeout(self.timer);
  self.timer = setTimeout(() => {
    self.last_updated = Date.now();
    updateCanvases(concept, col, row);
  }, 200);
}


// Update position selector dropdown
function updatePositionSelector() {
  const positions = Object.keys(current_model.representations);
  if (!positions.includes(current_position)) {
    if (current_model.short == 'SD3') current_position = 'transformer_blocks[12]';
    else if (current_model.short.includes('SDXL')) current_position = 'up_blocks[0]';
    else if (positions.includes('up_blocks[1]')) current_position = 'up_blocks[1]';
    else current_position = positions[0];
  }
  document.getElementById('position-to-use').innerHTML = '';  // clear old options
  positions.forEach(position => {
    // setup available positions (where the representations are extracted from)
    createElem('option', {
      value: position,
      textContent: position,
      selected: position == current_position
    }, document.getElementById('position-to-use'));
  });
}


// Update the model description at the top
function updateModelDescription() {
  const { m, n} = current_model.getShapes();
  const sim_elem = document.getElementById('similarity-measure');
  const sim_description = {
    'cosine': 'as the dot product normalized by the L2 norms',
    'cosine_centered': 'just as the cosine similarity, but with the mean of each representation subtracted before',
    'dot-product': 'as the dot product, normalized by the maximum absolute value',
    'manhattan': 'as the absolute difference, normalized by the maximum value and subtracted from 1',
    'euclidean': 'as the root of the sum of squared differences, normalized by the maximum value and subtracted from 1',
    'chebyshev': 'as the maximum absolute difference, normalized by the maximum value and subtracted from 1',
    'rel-l2-norm': 'as the relative L2 norm between the two representations',
  }[sim_elem.value];
  document.getElementById('model-description').innerHTML = `The examples above are the output for <a href="https://huggingface.co/${current_model.name}">${current_model.name}</a> at position ${current_position} in the diffusion unet, which has a dimension of (${m},${n},${n}), i.e. ${m} channels, and ${n}×${n} spatial resolution for the default image size of ${current_model.default_image_shape.slice(1).join('×')}. ${sim_elem.options[sim_elem.selectedIndex].textContent} similarity is calculated ${sim_description}.`;
};


// get mouse/touch position on canvas
function getMousePos(canvas, event) {
  const { n } = current_model.getShapes();
  if (event.touches) event = event.touches[0];  // handle touch events for mobile
  const rect = canvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;
  const col = Math.floor(2 * x / canvas.width * n);
  const row = Math.floor(2 * y / canvas.width * n);
  return { col, row };
}


// Get cached images from the server and their available representations
function getCachedImages() {
  return fetch(`${data_base_url}cached_images.json`)
    .then(response => response.json())
    .then(data => {
      Object.assign(available_concepts, data.cached_images);
      const images_container = document.getElementById('image-to-use');
      images_container.innerHTML = '';
      Object.keys(available_concepts).forEach((name, i) => {
        createElem('img', { src: `${data_base_url}images_small/${name}.jpg`, onclick: () => {concepts.push(setupConcept(name)); updateCanvasesWithLastClicked();} }, images_container);
      });
      return data.cached_images;
    });
}


// Initialize the app
function init() {
  // load available prompts/concepts
  const cached_images_promise = getCachedImages();

  // load available models
  const models_promise = fetch(`${data_base_url}model_info.json`)
    .then(response => response.json())
    .then(data => {
      available_models.push(...data);
      available_models.forEach(model => {
        createElem('option', { value: model.short, textContent: model.short }, document.getElementById('model-to-use'));
        model.getShapes = () => ({ m: model.representations[current_position].channels, n: model.representations[current_position].spatial })
      });
      current_model = available_models[0];
      updatePositionSelector();
      updateModelDescription();
    })
    .catch(error => console.error('Error while fetching models info:', error));

  // initialize concepts
  Promise.all([cached_images_promise, models_promise])
    .then(() => {
      // update canvases on load with a delay
      setTimeout(updateCanvasesWithLastClicked, 100);
    });

    // setup model change event listener
    document.getElementById('model-to-use').addEventListener('change', event => {
      const prev_hspace_spatial = current_model.getShapes().n;
  
      // update current model, position selector, and description
      current_model = available_models.find(x => x.short === event.target.value);
      updatePositionSelector();
      updateModelDescription();
  
      // update last clicked if spatial resolution changed
      const { n } = current_model.getShapes();
      if (n !== prev_hspace_spatial) {
        last_clicked.col = Math.floor(last_clicked.col * n / prev_hspace_spatial);
        last_clicked.row = Math.floor(last_clicked.row * n / prev_hspace_spatial);
      }

      // update concepts and canvases
      concepts.forEach(concept => concept.update());
      updateCanvasesWithLastClicked();
    });

  // setup position change event listener
  document.getElementById('position-to-use').addEventListener('change', event => {
    const prev_hspace_spatial = current_model.getShapes().n;

    // update current position and description
    current_position = event.target.value;
    updateModelDescription();

    // update last clicked if spatial resolution changed
    const { n } = current_model.getShapes();
    if (n !== prev_hspace_spatial) {
      last_clicked.col = Math.floor(last_clicked.col * n / prev_hspace_spatial);
      last_clicked.row = Math.floor(last_clicked.row * n / prev_hspace_spatial);
    }

    // update concepts and canvases
    concepts.forEach(concept => concept.update());
    updateCanvasesWithLastClicked();
  });

  // setup similarity measure change event listener
  document.getElementById('similarity-measure').addEventListener('change', () => {
    updateModelDescription();
    updateCanvasesWithLastClicked();
  });

  // setup preset buttons
  setupPresetButtons();

  // setup info buttons
  setupInfoButtons();

  // reload on resize to update canvas size
  window.addEventListener('resize', () => {
      concepts.forEach(concept => {
        const canvas_size = concept.image_ctx.canvas.parentElement.clientWidth * 2;
        concept.image_canvas.width = canvas_size;
        concept.image_canvas.height = canvas_size;
        concept.tile_canvas.width = canvas_size;
        concept.tile_canvas.height = canvas_size;
      });
      updateCanvasesWithLastClicked();
  });
}
  
// Define preset configurations with direct mapping to button IDs
const presets = {
  'preset-positional': {
    model: 'SD1.5',
    position: 'up_blocks[0]',
    similarity: 'cosine',
    images: ['41afce3feb7d4230265726762545efe6','3f61de85ac98cf25e584aeddbfd1a1e4'],
    click: {
      "concept_index": 0,
      "col": 13,
      "row": 2
    }
  },
  'preset-corner': {
    model: 'SD2.1',
    position: 'mid_block',
    similarity: 'cosine',
    images: ['6b4bafdb74497c398085348fcc52312b','4178335c13a489c1238e66b9ce720443'],
    click: {
      "concept_index": 1,
      "col": 11,
      "row": 0
    }
  },
  'preset-norm': {
    model: 'SD-Turbo',
    position: 'up_blocks[1]',
    similarity: 'rel-l2-norm',
    images: ['41afce3feb7d4230265726762545efe6','6b4bafdb74497c398085348fcc52312b'],
    click: {
      "concept_index": 0,
      "col": 0,
      "row": 0
    }
  }
};

function clearPresetSelection() {
  Object.keys(presets).forEach(btnId => {
    document.getElementById(btnId).classList.remove('active');
  });
}

// Setup preset buttons functionality
function setupPresetButtons() {
  // Get DOM elements
  const modelSelect = document.getElementById('model-to-use');
  const positionSelect = document.getElementById('position-to-use');
  const similaritySelect = document.getElementById('similarity-measure');
  
  // Add click event listeners to preset buttons
  Object.entries(presets).forEach(([id, preset]) => {
    const button = document.getElementById(id);
    if (!button) return console.error(`Did not find button with id ${id}`);

    button.addEventListener('click', () => {
      // Trigger change event to update model
      modelSelect.value = preset.model;
      modelSelect.dispatchEvent(new Event('change'));

      // Trigger change event to update position
      positionSelect.value = preset.position;
      positionSelect.dispatchEvent(new Event('change'));

      // Trigger change event to update similarity
      similaritySelect.value = preset.similarity;
      similaritySelect.dispatchEvent(new Event('change'));

      // Set the images
      while (concepts.length > 0) concepts[0].delete();
      preset.images.forEach(image => {
        const concept = setupConcept(image);
        if (concept) concepts.push(concept);
      });

      // update the last clicked (selected) tile
      last_clicked = {
        concept_id: concepts[preset.click.concept_index].id,
        col: preset.click.col,
        row: preset.click.row
      };
      updateCanvasesWithLastClicked();

      // Add active class to clicked button
      clearPresetSelection();
      button.classList.add('active');

    });
  });
  
  // Check settings when any selector changes
  [modelSelect, positionSelect, similaritySelect].forEach(select => {
    select.addEventListener('change', clearPresetSelection);
  });
}

// Setup info buttons functionality
function setupInfoButtons() {
  const infoButtons = document.querySelectorAll('.info-button');
  let activePopup = null;
  
  // Function to close any open popup
  function closePopup() {
    if (activePopup) {
      activePopup.remove();
      activePopup = null;
    }
  }
  
  // Add click event listeners to info buttons
  infoButtons.forEach(button => {
    button.addEventListener('click', (event) => {
      event.stopPropagation(); // Prevent event from bubbling up
      
      // If this button already has an open popup, just close it
      if (activePopup) {
        if (activePopup._sourceButton === button) {
          closePopup();
          return;
        } else {
          closePopup();
        }
      }
      
      // Get the info text from the data attribute
      const infoText = button.getAttribute('data-info');
      if (!infoText) return console.error(`No info text found for button ${button.id}`);
      
      // Create the popup
      const popup = document.createElement('div');
      popup.className = 'info-popup';
      popup.textContent = infoText;
      popup._sourceButton = button; // Store reference to source button
      
      // Add to body instead of as child of button
      document.body.appendChild(popup);
      
      // Position off-screen first to get dimensions
      popup.style.visibility = 'hidden';
      
      // Position the popup near the button after a brief delay to ensure rendering
      setTimeout(() => {
        const buttonRect = button.getBoundingClientRect();
        const popupRect = popup.getBoundingClientRect();
        
        // Calculate position (centered below the button by default)
        let left = buttonRect.left + (buttonRect.width / 2) - (popupRect.width / 2);
        let top = buttonRect.bottom + 8; // 8px gap
        let pointerClass = 'pointer-top'; // Default pointer at top
        
        // Adjust if going off screen
        if (left < 10) left = 10;
        if (left + popupRect.width > window.innerWidth - 10) {
          left = window.innerWidth - 10 - popupRect.width;
        }
        
        // If popup would go below viewport, position it above the button
        if (top + popupRect.height > window.innerHeight - 10) {
          top = buttonRect.top - popupRect.height - 8;
          pointerClass = 'pointer-bottom'; // Switch to bottom pointer
        }
        
        // Apply position and pointer class
        popup.style.left = `${left}px`;
        popup.style.top = `${top}px`;
        popup.classList.add(pointerClass);
        popup.style.visibility = 'visible';
      }, 10);
      
      // Set as active popup
      activePopup = popup;
    });
  });
  
  // Close popup when clicking anywhere else
  document.addEventListener('click', (event) => {
    if (activePopup && !event.target.closest('.info-button')) {
      closePopup();
    }
  });
  
  // Close popup when pressing Escape
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && activePopup) {
      closePopup();
    }
  });
}

// start the app
init();
