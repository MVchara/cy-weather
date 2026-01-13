import type { WeatherResponse, ForecastResponse } from '../types/weather';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'https://cy-weather-nine.vercel.app/api/';

/**
 * Récupère la météo actuelle pour une ville donnée
 * @param city Nom de la ville
 * @param countryCode Code pays ISO optionnel (ex: FR, US)
 * @returns Données météo actuelles
 */
export async function getCurrentWeather(
    city: string,
    countryCode?: string
): Promise<WeatherResponse> {
    const params = new URLSearchParams({ city });
    if (countryCode) {
        params.append('country_code', countryCode);
    }

    const response = await fetch(`${API_BASE_URL}/weather/current?${params}`);

    if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: 'Erreur réseau' }));
        throw new Error(error.detail || `Erreur HTTP: ${response.status}`);
    }

    return response.json();
}

/**
 * Récupère les prévisions météo sur 7 jours pour une ville donnée
 * @param city Nom de la ville
 * @param countryCode Code pays ISO optionnel (ex: FR, US)
 * @returns Prévisions sur 7 jours
 */
export async function getWeatherForecast(
    city: string,
    countryCode?: string
): Promise<ForecastResponse> {
    const params = new URLSearchParams({ city });
    if (countryCode) {
        params.append('country_code', countryCode);
    }

    const response = await fetch(`${API_BASE_URL}/weather/forecast?${params}`);

    if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: 'Erreur réseau' }));
        throw new Error(error.detail || `Erreur HTTP: ${response.status}`);
    }

    return response.json();
}
