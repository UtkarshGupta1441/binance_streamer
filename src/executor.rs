use hmac::{Hmac, Mac};
use sha2::Sha256;
use std::time::{SystemTime, UNIX_EPOCH};
use pyo3::prelude::*;

type HmacSha256 = Hmac<Sha256>;

/// Order executor for Binance API with HMAC-SHA256 signing
#[pyclass]
pub struct OrderExecutor {
    api_key: String,
    secret_key: String,
    base_url: String,
    testnet: bool,
}

#[pymethods]
impl OrderExecutor {
    #[new]
    #[pyo3(signature = (api_key, secret_key, testnet=true))]
    pub fn new(api_key: String, secret_key: String, testnet: bool) -> Self {
        let base_url = if testnet {
            "https://testnet.binance.vision/api/v3".to_string()
        } else {
            "https://api.binance.com/api/v3".to_string()
        };

        Self {
            api_key,
            secret_key,
            base_url,
            testnet,
        }
    }

    /// Sign a query string using HMAC-SHA256
    pub fn sign(&self, query_string: &str) -> String {
        let mut mac = HmacSha256::new_from_slice(self.secret_key.as_bytes())
            .expect("HMAC can take key of any size");
        mac.update(query_string.as_bytes());
        hex::encode(mac.finalize().into_bytes())
    }

    /// Get current timestamp in milliseconds
    pub fn get_timestamp(&self) -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64
    }

    /// Execute a market order
    #[pyo3(signature = (symbol, side, quantity, dry_run=true))]
    pub fn execute_market_order(
        &self,
        symbol: &str,
        side: &str,
        quantity: f64,
        dry_run: bool,
    ) -> PyResult<String> {
        let timestamp = self.get_timestamp().to_string();
        let qty_str = format!("{:.8}", quantity);
        
        let query_string = format!(
            "symbol={}&side={}&type=MARKET&quantity={}&timestamp={}",
            symbol.to_uppercase(),
            side.to_uppercase(),
            qty_str,
            timestamp
        );

        let signature = self.sign(&query_string);
        let url = format!("{}/order?{}&signature={}", self.base_url, query_string, signature);

        if dry_run {
            return Ok(format!(
                "DRY RUN: Would execute {} {} {} on {} (URL: {})",
                side.to_uppercase(),
                quantity,
                symbol.to_uppercase(),
                if self.testnet { "TESTNET" } else { "PRODUCTION" },
                url
            ));
        }

        // Actual execution using blocking reqwest
        let client = reqwest::blocking::Client::new();
        let response = client
            .post(&url)
            .header("X-MBX-APIKEY", &self.api_key)
            .send()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Request failed: {}", e)))?;

        let status = response.status();
        let body = response.text().unwrap_or_default();

        if status.is_success() {
            Ok(format!("SUCCESS: {}", body))
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Order failed ({}): {}",
                status, body
            )))
        }
    }

    /// Execute a limit order
    #[pyo3(signature = (symbol, side, quantity, price, dry_run=true))]
    pub fn execute_limit_order(
        &self,
        symbol: &str,
        side: &str,
        quantity: f64,
        price: f64,
        dry_run: bool,
    ) -> PyResult<String> {
        let timestamp = self.get_timestamp().to_string();
        let qty_str = format!("{:.8}", quantity);
        let price_str = format!("{:.8}", price);
        
        let query_string = format!(
            "symbol={}&side={}&type=LIMIT&timeInForce=GTC&quantity={}&price={}&timestamp={}",
            symbol.to_uppercase(),
            side.to_uppercase(),
            qty_str,
            price_str,
            timestamp
        );

        let signature = self.sign(&query_string);
        let url = format!("{}/order?{}&signature={}", self.base_url, query_string, signature);

        if dry_run {
            return Ok(format!(
                "DRY RUN: Would execute LIMIT {} {} {} @ {} on {}",
                side.to_uppercase(),
                quantity,
                symbol.to_uppercase(),
                price,
                if self.testnet { "TESTNET" } else { "PRODUCTION" }
            ));
        }

        let client = reqwest::blocking::Client::new();
        let response = client
            .post(&url)
            .header("X-MBX-APIKEY", &self.api_key)
            .send()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Request failed: {}", e)))?;

        let status = response.status();
        let body = response.text().unwrap_or_default();

        if status.is_success() {
            Ok(format!("SUCCESS: {}", body))
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Order failed ({}): {}",
                status, body
            )))
        }
    }

    /// Check if using testnet
    pub fn is_testnet(&self) -> bool {
        self.testnet
    }

    /// Get the base URL being used
    pub fn get_base_url(&self) -> &str {
        &self.base_url
    }
}
