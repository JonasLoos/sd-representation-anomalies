use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use std::collections::HashMap;
use half::f16;
use web_sys::{js_sys, Request, RequestInit, RequestMode, Response, console};
use std::cell::RefCell;
use js_sys::{ArrayBuffer, Uint8Array};
use console_error_panic_hook;
use std::panic;
use ndarray::{s, Array1, Array2, ArrayView1, Axis, Zip};
use std::sync::Arc;



macro_rules! jserr {() => {|e| JsValue::from_str(&format!("WASM: {:#?}", e))};}
macro_rules! jsnone {() => {JsValue::from_str(&format!("WASM: unexpected None in {}:{}:{}", file!(), line!(), column!()))};}


// cache to store representations and means
thread_local! {
    static GLOBAL_MAP: RefCell<HashMap<String, Arc<(Array2<f32>,Array1<f32>,Array1<f32>)>>> = RefCell::new(HashMap::new());
}


// calculate similarities between one pixel of a base representation and all pixels of a second representation
#[wasm_bindgen]
pub fn calc_similarities(
    func: String,
    repr1_str: String,
    repr2_str: String,
    row: usize,
    col: usize,
) -> Result<Vec<f32>, JsValue> {

    // better error messages in the console
    console_error_panic_hook::set_once();

    // get representations from cache
    GLOBAL_MAP.with(|map| {
        let reprs = map.borrow();
        let arc_data1 = reprs.get(&repr1_str).ok_or_else(|| JsValue::from_str("loading"))?;
        let arc_data2 = reprs.get(&repr2_str).ok_or_else(|| JsValue::from_str("loading"))?;
        let (repr1, repr2, means1_full, means2_full, norms1, norms2) = (&arc_data1.0, &arc_data2.0, &arc_data1.1, &arc_data2.1, &arc_data1.2, &arc_data2.2);
        let n = (repr1.shape()[0] as f32).sqrt() as usize;

        // calculate similarities
        let a: ArrayView1<f32> = repr1.slice(s![row*n+col,..]);
        let mut similarities: Vec<f32> = match func.as_str() {
            "cosine" => repr2
                .axis_iter(Axis(0))
                .enumerate()
                .map(|(index, b)| b.dot(&a) / (norms1[[row*n+col]] * norms2[[index]] + 1e-10))
                .collect::<Vec<_>>(),
            "cosine_centered" => {
                let means = Zip::from(means1_full).and(means2_full).map_collect(|&mean1, &mean2| ((mean1 + mean2) / 2.0));
                let norm1_centered = Zip::from(a).and(means.view()).fold(0.0, |acc, &ai, &mean| acc + (ai - mean).powi(2)).sqrt();
                let norms2_centered = repr2.axis_iter(Axis(0)).map(|b| (Zip::from(b).and(means.view()).fold(0.0, |acc, &bi, &mean| acc + (bi - mean).powi(2))).sqrt()).collect::<Vec<_>>();
                repr2
                    .axis_iter(Axis(0))
                    .enumerate()
                    .map(|(index, b)| Zip::from(a).and(b).and(means.view()).fold(0.0, |acc, &ai, &bi, &mean| acc + (ai - mean) * (bi - mean)) / (norm1_centered * norms2_centered[index] + 1e-10))
                    .collect::<Vec<_>>()
            },
            "dot-product" => {
                let dot_product = repr2.axis_iter(Axis(0)).map(|b| b.dot(&a)).collect::<Vec<_>>();
                let max_dot_product = dot_product.iter().fold(0.0f32, |acc, &x| acc.max(x.abs())) + 1e-10;
                dot_product.iter().map(|&dp| dp / max_dot_product).collect::<Vec<_>>()
            },
            "manhattan" => repr2
                .axis_iter(Axis(0))
                .map(|b| Zip::from(a).and(b).fold(0.0, |acc, &ai, &bi| acc + (ai - bi).abs()))
                .collect::<Vec<_>>(),
            "euclidean" => repr2
                .axis_iter(Axis(0))
                .map(|b| Zip::from(a).and(b).fold(0.0, |acc, &ai, &bi| acc + (ai - bi).powi(2)).sqrt())
                .collect::<Vec<_>>(),
            "chebyshev" => repr2
                .axis_iter(Axis(0))
                .map(|b| Zip::from(a).and(b).fold(0.0, |acc: f32, &ai, &bi| acc.max((ai - bi).abs())))
                .collect::<Vec<_>>(),
            "rel-l2-norm" => {
                let diffs = norms2.iter().map(|&norm| norm - norms1[[row*n+col]]).collect::<Vec<_>>();
                let max_abs_diff = diffs.iter().fold(0.0f32, |acc, &x| acc.max(x.abs())) + 1e-10;
                diffs.iter().map(|&diff| diff / max_abs_diff).collect::<Vec<_>>()
            },
            _ => panic!("Unknown similarity function"),
        };

        // normalize distances
        if func == "euclidean" || func == "manhattan" || func == "chebyshev" {
            let max_distance = *similarities.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).ok_or(jsnone!())?;
            for distance in similarities.iter_mut() {
                *distance = 1.0 - (*distance / max_distance);
            }
        }

        Ok(similarities)
    })
}


// fetch representation from url and store it in cache
#[wasm_bindgen]
pub async fn fetch_repr(url: String, n: usize, m: usize) -> Result<(), JsValue> {
    console_error_panic_hook::set_once();  // better error messages in the console

    // if the representation is already fetched, return
    if GLOBAL_MAP.with(|map| map.borrow().contains_key(&url)) {
        return Ok(());
    }

    // initialize fetch request
    let opts = RequestInit::new();
    opts.set_method("GET");
    opts.set_mode(RequestMode::Cors);
    let request = Request::new_with_str_and_init(&url, &opts)?;

    // fetch representation
    let global = js_sys::global().unchecked_into::<web_sys::WorkerGlobalScope>();
    let resp: Response = match JsFuture::from(global.fetch_with_request(&request)).await {
        Ok(value) => value.dyn_into().map_err(jserr!())?,
        Err(e) => return Err(JsValue::from_str(&format!("Failed to fetch representation ({}): {:#?}", url, e)))
    };

    // convert response to float16 vector
    let buffer: ArrayBuffer = JsFuture::from(resp.array_buffer()?).await?.dyn_into().map_err(jserr!())?;
    if buffer.byte_length() % 2 != 0 {
        return Err(JsValue::from_str(&format!("Buffer length is not a multiple of 2 (for float16): {}", url)));
    }
    let bytes = Uint8Array::new(&buffer).to_vec();
    let float16_data: Vec<f16> = bytes.chunks_exact(2).map(|chunk| f16::from_le_bytes([chunk[0], chunk[1]])).collect();

    // convert float16 vector to Array4<f32>
    let representations = match Array2::from_shape_vec((n*n, m), float16_data.iter().map(|&x| f32::from(x)).collect()) {
        Ok(repr) => repr,
        Err(e) => {
            return Err(JsValue::from_str(format!("Failed to convert float16 vector (len {}, {}) to Array3<f32> with shape ({}, {}): {:#?}", float16_data.len(), url, n*n, m, e).as_str()))
        }
    };

    // store representations and means and norms in cache
    let means = representations.mean_axis(Axis(0)).ok_or(jsnone!())?;
    let norms = representations.mapv(|x| x.powi(2)).sum_axis(Axis(1)).mapv(f32::sqrt);
    GLOBAL_MAP.with(|map| {
        map.borrow_mut().insert(url.to_string(), Arc::new((representations, means, norms)));
    });

    Ok(())
}
