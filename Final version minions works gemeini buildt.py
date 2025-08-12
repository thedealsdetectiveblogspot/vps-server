import time
import random
import threading
import smtplib
import json
import numpy as np
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta, date
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (NoSuchElementException,
                                        ElementClickInterceptedException,
                                        ElementNotInteractableException,
                                        TimeoutException,
                                        StaleElementReferenceException,
                                        WebDriverException,
                                        MoveTargetOutOfBoundsException)
import requests
from pathlib import Path
import hashlib
import platform
import psutil
from fake_useragent import UserAgent
import ssl
from urllib3.exceptions import InsecureRequestWarning
import socket
import traceback
import logging
import os
import sys

# --- NEW Imports for Firebase Email System ---
import firebase_admin
from firebase_admin import credentials, firestore

# Disable SSL warnings for requests (for proxy fetching/testing)
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("neuro_bot_illusion_lite.log", encoding='utf-8'),
                        logging.StreamHandler(sys.stdout)
                    ])
if platform.system() == "Windows":
    for handler in logging.root.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            if hasattr(handler.stream, 'reconfigure'):
                handler.stream.reconfigure(encoding='utf-8')


# --- Constants ---
NEURO_VERSION = "2.6.3-IllusionLite" # Updated version for tracking
MAX_THREADS_DEFAULT = 5 # Will be varied by Overmind
MIN_SESSION_DURATION = 25
MAX_SESSION_DURATION = 70
AD_CLICK_PROBABILITY = 0.7
MAX_AD_CLICKS = 5
MAX_RETRIES = 2
RETRY_DELAY_BASE = 5
PROXY_REFRESH_INTERVAL = 3600

# --- Illusion Engine Specific Constants ---
# This will be initialized in NeuroThreadManager._update_overmind_params at start
# and then daily. We set a default here just in case.
DAILY_VISIT_QUOTA = random.randint(500, 1000)


# --- Proxy Constants ---
TRUSTED_PROXIES = [] # Populate this list or use environment variable
PROXY_TEST_URLS = ["http://example.com", "http://google.com"]
PROXY_TEST_TIMEOUT = 8
PROXY_MAX_ATTEMPTS = 3

# --- Ad Detection Selectors ---
AD_SELECTORS = list(set([
    '.ad', '.ads', '.advertisement', '[class*="ad-"]',
    '[id*="ad-"]', '.banner-ad', '.ad-banner', '.ad-wrapper',
    '.ad-container', '.ad-unit', '.advert',
    '.adslot', '.ad-placeholder', '.adlink', '.adbox',
    '.product-ad', '.deal-ad', '.promo-box', '.sponsored',
    '.recommended', '.sponsored-content', '.partner-box',
    '[data-ad]', '[data-ad-id]', '[data-ad-target]',
    '[data-ad-client]', '[data-ad-slot]', '[data-ad-type]',
    'iframe[src*="ads"]', 'iframe[src*="adserver"]',
    'iframe[src*="doubleclick"]', 'iframe[src*="googleadservices"]',
    '.widget_sp_image', '.imagewidget', '.promoted-link',
    '.affiliate-link', '.partner-link'
]))

# --- NeuroPersonalityCore ---
class NeuroPersonalityCore:
    ARCHETYPES = {
        'ad_clicker': { # High ad interaction
            'traits': {'neuroticism': 0.6, 'openness': 0.5, 'extraversion': 0.4},
            'behavior': {'scroll_speed': 1.3, 'click_accuracy': 0.8, 'ad_avoidance': 0.1, 'reading_pattern': 'skimming'},
            'ad_behavior': {'click_probability': 0.9, 'max_clicks': MAX_AD_CLICKS},
            'session_duration_modifier': (0.9, 1.2)
        },
        'shopper': { # High ad interaction, product-focused
            'traits': {'extraversion': 0.6, 'openness': 0.5, 'neuroticism': 0.4},
            'behavior': {'scroll_speed': 1.2, 'click_accuracy': 0.8, 'ad_avoidance': 0.2, 'reading_pattern': 'scanning'},
            'ad_behavior': {'click_probability': 0.75, 'max_clicks': 4},
            'session_duration_modifier': (1.0, 1.3)
        },
        'casual_reader': { # Lower ad interaction, content-focused
            'traits': {'neuroticism': 0.3, 'conscientiousness': 0.7, 'agreeableness': 0.6},
            'behavior': {'scroll_speed': 1.0, 'click_accuracy': 0.6, 'ad_avoidance': 0.5, 'reading_pattern': 'linear'},
            'ad_behavior': {'click_probability': 0.3, 'max_clicks': 2},
            'session_duration_modifier': (1.0, 1.2)
        },
        'researcher': { # Very low ad interaction, deep content
            'traits': {'openness': 0.9, 'conscientiousness': 0.8, 'extraversion': 0.2},
            'behavior': {'scroll_speed': 0.8, 'click_accuracy': 0.9, 'ad_avoidance': 0.7, 'reading_pattern': 'deep'},
            'ad_behavior': {'click_probability': 0.1, 'max_clicks': 1},
            'session_duration_modifier': (1.1, 1.4)
        },
        'bouncer': { # Very short visit, high bounce
            'traits': {'neuroticism': 0.5, 'openness': 0.2, 'extraversion': 0.3},
            'behavior': {'scroll_speed': 1.8, 'click_accuracy': 0.5, 'ad_avoidance': 0.9, 'reading_pattern': 'none'},
            'ad_behavior': {'click_probability': 0.01, 'max_clicks': 0},
            'session_duration_modifier': (0.1, 0.25)
        },
        'skimmer': { # Fast reads, surface interaction, multiple pages if possible
            'traits': {'openness': 0.7, 'extraversion': 0.6, 'conscientiousness': 0.4},
            'behavior': {'scroll_speed': 1.7, 'click_accuracy': 0.7, 'ad_avoidance': 0.4, 'reading_pattern': 'skimming'},
            'ad_behavior': {'click_probability': 0.2, 'max_clicks': 1},
            'session_duration_modifier': (0.8, 1.1)
        },
        'idle_reader': { # Loads page, simulates idle presence, few interactions
            'traits': {'neuroticism': 0.2, 'conscientiousness': 0.5, 'agreeableness': 0.7},
            'behavior': {'scroll_speed': 0.9, 'click_accuracy': 0.6, 'ad_avoidance': 0.6, 'reading_pattern': 'linear_slow_start'},
            'ad_behavior': {'click_probability': 0.05, 'max_clicks': 1},
            'session_duration_modifier': (1.2, 1.8)
        }
    }

    @staticmethod
    def generate_personality():
        if random.random() < AD_CLICK_PROBABILITY:
            archetype_candidate = random.choice(['ad_clicker', 'shopper'])
        else:
            archetype_candidate = random.choice(['casual_reader', 'researcher', 'bouncer', 'skimmer', 'idle_reader'])

        if archetype_candidate == 'bouncer' and random.random() > 0.3:
            archetype_candidate = random.choice(['casual_reader', 'researcher', 'skimmer', 'idle_reader'])

        base = NeuroPersonalityCore.ARCHETYPES[archetype_candidate]

        traits = {
            'cognitive': {
                'openness': np.clip(base['traits'].get('openness', 0.5) + random.uniform(-0.2, 0.2), 0, 1),
                'curiosity': random.uniform(0, 1), 'learning_speed': random.uniform(0.3, 0.9)},
            'emotional': {
                'neuroticism': np.clip(base['traits'].get('neuroticism', 0.5) + random.uniform(-0.2, 0.2), 0, 1),
                'mood_stability': random.uniform(0.2, 0.8), 'stress_response': random.uniform(0.1, 0.9)},
            'social': {
                'extraversion': np.clip(base['traits'].get('extraversion', 0.5) + random.uniform(-0.2, 0.2), 0, 1),
                'agreeableness': np.clip(base['traits'].get('agreeableness', 0.5) + random.uniform(-0.2, 0.2), 0, 1),
                'trust': random.uniform(0.2, 0.8)},
            'motor': {
                'coordination': random.uniform(0.4, 0.9), 'speed_variability': random.uniform(0.1, 0.7),
                'accuracy': np.clip(base['behavior'].get('click_accuracy', 0.5) + random.uniform(-0.2, 0.2), 0, 1)}
        }
        behavior = {
            'scroll_speed': base['behavior'].get('scroll_speed', 1.0) * random.uniform(0.8, 1.2),
            'ad_avoidance': base['behavior'].get('ad_avoidance', 0.5) * random.uniform(0.7, 1.3),
            'reading_pattern': base['behavior'].get('reading_pattern', 'linear'),
            'attention_span': random.uniform(5, 45) * (0.5 if archetype_candidate == 'bouncer' else 1.0),
            'error_rate': 1 - traits['motor']['accuracy'],
            'device_preference': random.choice(['desktop', 'mobile', 'tablet']),
            'ad_click_probability': np.clip(base['ad_behavior']['click_probability'] * random.uniform(0.8, 1.2), 0, 1),
            'max_ad_clicks': min(MAX_AD_CLICKS, base['ad_behavior']['max_clicks'] + random.randint(-1, 1)),
            'session_duration_modifier': base.get('session_duration_modifier', (0.9, 1.1))
        }

        possible_referrers = [
            None, "https://www.google.com/", "https://www.bing.com/",
            "https://duckduckgo.com/", "https://t.co/", "https://www.facebook.com/",
        ]
        chosen_referrer = random.choice(possible_referrers)
        if archetype_candidate == 'bouncer' and random.random() < 0.7:
            chosen_referrer = None

        fingerprint = {
            'browser_taints': NeuroPersonalityCore.generate_browser_taints(),
            'device_profile': NeuroPersonalityCore.generate_device_profile(behavior['device_preference']),
            'network_profile': NeuroPersonalityCore.generate_network_profile(),
            'referrer_url': chosen_referrer
        }
        return {'archetype': archetype_candidate, 'traits': traits, 'behavior': behavior, 'fingerprint': fingerprint,
                'state': 'initializing', 'cognitive_load': 0.0,
                'session_goals': random.sample(['read_content', 'find_deals', 'social_interact', 'time_waste'], k=random.randint(1,2)),
                'ad_clicks': 0}

    @staticmethod
    def generate_browser_taints():
        return [f"canvas_noise:{random.randint(1, 10)}",
                f"audio_ctx_hash:{hashlib.md5(str(random.random()).encode()).hexdigest()[:8]}",
                f"webgl_vendor:{random.choice(['Intel Inc.', 'NVIDIA Corporation', 'AMD', 'Google Inc. (ANGLE)'])}",
                f"timezone_offset:{random.choice(range(-12*60, 14*60+1, 15))}",
                f"font_hash:{hashlib.sha256(str(random.random()).encode()).hexdigest()[:16]}"]

    @staticmethod
    def generate_device_profile(device_type):
        if device_type == 'mobile':
            return {'type': 'mobile', 'os': random.choice(['iOS 17.1', 'Android 13', 'iOS 16.5', 'Android 14']),
                    'screen': f"{random.choice([375, 390, 412, 414])}x{random.choice([667, 812, 844, 852, 896])}",
                    'touch': True, 'pixel_ratio': random.choice([2, 3])}
        elif device_type == 'tablet':
            return {'type': 'tablet', 'os': random.choice(['iPadOS 17', 'Android 13 Tablet', 'iPadOS 16.2']),
                    'screen': f"{random.choice([768, 810, 834])}x{random.choice([1024, 1080, 1112, 1180])}",
                    'touch': True, 'pixel_ratio': random.choice([1, 2])}
        else: # desktop
            return {'type': 'desktop', 'os': random.choice(['Windows 10', 'Windows 11', 'Mac OS X 10.15', 'Mac OS X 13.0']),
                    'screen': f"{random.choice([1280, 1366, 1440, 1536, 1920])}x{random.choice([720, 768, 800, 864, 900, 1080])}",
                    'touch': False, 'pixel_ratio': random.choice([1, 1.25, 1.5, 2])}

    @staticmethod
    def generate_network_profile():
        return {'latency': random.randint(10, 350),
                'bandwidth': random.choice(['DSL', 'Cable', 'Fiber', '4G LTE', '5G', 'Satellite']),
                'stability': random.uniform(0.6, 0.99)}

class AutoPilot:
    def __init__(self, driver, personality):
        self.driver = driver
        self.personality = personality
        self.state_machine = {
            'initializing': self._state_initializing, 'browsing': self._state_browsing,
            'reading': self._state_reading, 'interacting': self._state_interacting,
            'distracted': self._state_distracted, 'ad_scanning': self._state_ad_scanning,
            'idling': self._state_idling, 'bouncing': self._state_bouncing
            }
        self.current_state = 'initializing'
        self.last_state_change = time.time()
        self.cognitive_load = 0.0
        self.attention_span = personality['behavior']['attention_span']
        self.reading_progress = 0
        self.behavior_log = []
        self.ad_elements = []
        self.last_ad_scan = 0
        self.idle_cycles = 0

    def step(self):
        try:
            if self.personality['archetype'] == 'bouncer' and self.current_state != 'bouncing':
                if self.current_state == 'initializing' and (time.time() - self.last_state_change) > random.uniform(0.5,2):
                     self._transition_to('bouncing')
                elif self.current_state != 'initializing':
                     self._transition_to('bouncing')

            self._evaluate_state()
            if time.time() - self.last_ad_scan > random.uniform(10, 30):
                self._scan_for_ads()
                self.last_ad_scan = time.time()

            if self.current_state in self.state_machine: self.state_machine[self.current_state]()
            else:
                logging.warning(f"Unknown state: {self.current_state}, defaulting to browsing.")
                self._transition_to('browsing')
                self.state_machine['browsing']()
            self._update_cognitive_load()
        except Exception as e:
            self.behavior_log.append({'time': datetime.now().isoformat(), 'event': 'autopilot_step_error',
                                      'details': str(e), 'state': self.current_state, 'stacktrace': traceback.format_exc()})
            logging.error(f"Error within AutoPilot step (State: {self.current_state}): {type(e).__name__} - {e}", exc_info=False)

    def _scan_for_ads(self):
        current_ads = []
        unique_ad_locations = {}
        for selector in AD_SELECTORS:
            try:
                elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                for el in elements:
                    try:
                        if not (el.is_enabled() and el.size['width'] > 10 and el.size['height'] > 10): continue
                        if el.is_displayed():
                            loc = el.location
                            pos_key = f"{loc['x']}_{loc['y']}_{el.size['width']}_{el.size['height']}"
                            if pos_key not in unique_ad_locations:
                                current_ads.append(el)
                                unique_ad_locations[pos_key] = el
                    except (StaleElementReferenceException, WebDriverException): continue
            except WebDriverException: continue
        self.ad_elements = current_ads
        if self.ad_elements:
             self.behavior_log.append({'time': datetime.now().isoformat(), 'event': 'ad_scan_completed',
                                   'ads_found_visible': len(self.ad_elements), 'state': self.current_state})

    def _state_ad_scanning(self):
        if (self.personality['ad_clicks'] >= self.personality['behavior']['max_ad_clicks'] or not self.ad_elements):
            self._transition_to('browsing'); return

        clickable_ads = []
        for ad_el in self.ad_elements:
            try:
                if ad_el.is_displayed() and ad_el.is_enabled(): clickable_ads.append(ad_el)
            except StaleElementReferenceException: continue
        if not clickable_ads: self._transition_to('browsing'); return

        if random.random() < self.personality['behavior']['ad_click_probability']:
            ad_to_click = random.choice(clickable_ads)
            ad_loc_before_click_val = None
            try: ad_loc_before_click_val = ad_to_click.location_once_scrolled_into_view
            except StaleElementReferenceException: logging.warning("Ad element for location stale.")

            try:
                logging.info(f"Personality {self.personality['archetype']} hovering over ad.")
                ActionChains(self.driver).move_to_element(ad_to_click).pause(random.uniform(2, 5)).perform()
            except Exception as e_hover: logging.warning(f"Error during pre-ad-click hover: {e_hover}")

            if self._human_click(ad_to_click):
                self.personality['ad_clicks'] += 1
                self.behavior_log.append({'time': datetime.now().isoformat(), 'event': 'ad_click',
                                          'ad_position': ad_loc_before_click_val,
                                          'total_ad_clicks_this_session': self.personality['ad_clicks']})
                logging.info(f"Clicked ad. Total ad clicks: {self.personality['ad_clicks']}")
                time.sleep(random.uniform(3, 8)) # Pause on ad page

                if len(self.driver.window_handles) > 1:
                    original_window = self.driver.current_window_handle
                    for handle in self.driver.window_handles:
                        if handle != original_window:
                            self.driver.switch_to.window(handle); time.sleep(random.uniform(0.5,1.0)); self.driver.close(); break
                    self.driver.switch_to.window(original_window)
                else:
                    try: self.driver.back(); time.sleep(random.uniform(0.5,1.0))
                    except WebDriverException as e: logging.warning(f"Error navigating back after ad: {e}")
                time.sleep(random.uniform(0.5, 1.5))
            else: logging.warning("Attempted ad click failed by _human_click.")
        self._transition_to('browsing')

    def _evaluate_state(self):
        now = time.time(); state_duration = now - self.last_state_change

        if self.personality['archetype'] == 'bouncer' and self.current_state not in ['bouncing', 'initializing']:
            self._transition_to('bouncing'); return

        if self.personality['archetype'] == 'idle_reader':
            if self.current_state not in ['initializing', 'idling'] and state_duration > random.uniform(10,20) and random.random() < 0.6:
                 self._transition_to('idling'); return
            elif self.current_state == 'idling' and self.idle_cycles > random.randint(2,4) and random.random() < 0.4:
                self._transition_to(random.choice(['browsing', 'reading'])); self.idle_cycles=0; return

        if (self.current_state != 'ad_scanning' and self.ad_elements and
            self.personality['ad_clicks'] < self.personality['behavior']['max_ad_clicks'] and
            random.random() < 0.35 and state_duration > random.uniform(3,10) ):
            self._transition_to('ad_scanning'); return

        if self.current_state == 'reading' and self.reading_progress >= 1.0:
            self._transition_to(random.choice(['browsing', 'interacting'])); return
        elif random.random() < (0.03 + self.cognitive_load * 0.05) and state_duration > 7:
            self._transition_to('distracted'); return

        if self.current_state not in ['bouncing', 'idling']:
            transition_probabilities = {
                'browsing': {'reading': 0.4, 'interacting': 0.2, 'distracted': 0.1, 'ad_scanning':0.15},
                'reading': {'browsing': 0.5, 'interacting': 0.15, 'distracted': 0.1},
                'interacting': {'browsing': 0.6, 'reading': 0.1, 'distracted': 0.15},
                'distracted': {'browsing': 0.7, 'reading': 0.1, 'interacting': 0.1}}
            current_probs_map = transition_probabilities.get(self.current_state, {'browsing': 0.8})
            choices = list(current_probs_map.keys()); weights = list(current_probs_map.values())
            if choices and sum(weights) > 0:
                if abs(sum(weights) - 1.0) > 0.001: weights = [w / sum(weights) for w in weights]
                next_state = random.choices(choices, weights=weights, k=1)[0]
                if next_state != self.current_state: self._transition_to(next_state)
            elif self.current_state != 'browsing': self._transition_to('browsing')

    def _transition_to(self, new_state):
        if self.current_state == new_state and (time.time() - self.last_state_change < 2): return
        logging.info(f"State change: {self.current_state} -> {new_state} (Load: {self.cognitive_load:.2f}, Arch: {self.personality['archetype']})")
        self.behavior_log.append({'time': datetime.now().isoformat(), 'event': 'state_change',
                                  'from': self.current_state, 'to': new_state, 'cognitive_load': self.cognitive_load})
        self.current_state = new_state; self.last_state_change = time.time()
        if new_state == 'reading': self.reading_progress = 0

    def _update_cognitive_load(self):
        change = random.uniform(-0.05, 0.08)
        if self.current_state == 'reading': change += random.uniform(0.01, 0.03)
        elif self.current_state == 'interacting': change += random.uniform(0.02, 0.05)
        elif self.current_state == 'ad_scanning': change += random.uniform(0.03, 0.06)
        self.cognitive_load = np.clip(self.cognitive_load + change, 0, 1)
        self.attention_span = self.personality['behavior']['attention_span'] * (1 - self.cognitive_load * 0.5)
        self.attention_span = max(3, self.attention_span)
        if self.cognitive_load > 0.9 and self.current_state != 'distracted' and random.random() < 0.7:
            logging.info(f"High cognitive load ({self.cognitive_load:.2f}), becoming distracted.")
            self._transition_to('distracted')

    def get_behavioral_delay(self):
        base_delays = {'browsing':0.3, 'reading':0.5, 'interacting':0.4, 'distracted':0.8,
                       'ad_scanning':0.2, 'initializing':0.5, 'idling':5.0, 'bouncing':0.1}
        speed_factor = max(0.1, self.personality['behavior']['scroll_speed'])
        cog_factor = np.clip(1 + (self.cognitive_load*random.uniform(-0.3,0.4)), 0.5, 1.5)
        base = base_delays.get(self.current_state, 0.4)
        delay = base / speed_factor * cog_factor * random.uniform(0.7, 1.3)
        if self.current_state == 'idling': return random.uniform(8, 25)
        if self.current_state == 'bouncing': return random.uniform(0.1, 0.5)
        return np.clip(delay, 0.05, 2.0)

    def _state_initializing(self):
        try: WebDriverWait(self.driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        except TimeoutException: logging.warning("Timeout on page body load.")
        self._human_scroll(random.uniform(0.1, 0.3)); time.sleep(random.uniform(0.5, 1.5))
        if self.personality['archetype'] == 'bouncer': self._transition_to('bouncing')
        elif self.personality['archetype'] == 'idle_reader': self._transition_to('idling')
        else: self._transition_to('browsing')

    def _state_browsing(self):
        weights = [0.4, 0.35, 0.15, 0.1] if self.personality['archetype'] == 'skimmer' else [0.6, 0.15, 0.15, 0.1]
        action = random.choices(['scroll', 'click_internal_link', 'hover', 'brief_pause'], weights=weights, k=1)[0]
        if action == 'scroll': self._human_scroll(random.uniform(0.05, 0.5) * random.choice([1,1,1,-1]))
        elif action == 'click_internal_link': self._find_and_interact(link_only=True)
        elif action == 'hover': self._random_hover()
        else: time.sleep(random.uniform(0.5, 1.5))

    def _state_reading(self):
        pattern = self.personality['behavior']['reading_pattern']
        if pattern == 'none': self.reading_progress = 1.0; self._transition_to('bouncing'); return
        speed = (random.uniform(0.02,0.05) if pattern == 'linear_slow_start' and self.reading_progress < 0.2 else
                 random.uniform(0.15,0.35) if pattern == 'skimming' or self.personality['archetype'] == 'skimmer' else
                 random.uniform(0.05,0.15))
        if pattern in ['linear', 'linear_slow_start']:
            self.reading_progress = min(1.0, self.reading_progress + speed)
            self._human_scroll(speed * random.uniform(0.3,0.6))
            if random.random() < 0.05:
                reg = random.uniform(-0.02,-0.01)
                self.reading_progress = max(0, self.reading_progress + reg); self._human_scroll(reg * random.uniform(0.2,0.4))
        elif pattern == 'scanning':
            if random.random() < 0.7: jump = random.uniform(0.15,0.35); self.reading_progress = min(1.0, self.reading_progress + jump); self._human_scroll(jump * 0.7)
            else: time.sleep(random.uniform(0.3,1))
        elif pattern == 'deep':
            deep_speed = random.uniform(0.03,0.1); self.reading_progress = min(1.0, self.reading_progress + deep_speed)
            self._human_scroll(deep_speed * 0.2); time.sleep(random.uniform(0.1,0.3))
        else: #skimming
            if random.random() < 0.8: jump = random.uniform(0.25,0.55); self.reading_progress = min(1.0, self.reading_progress + jump); self._human_scroll(jump*0.9)
            else: time.sleep(random.uniform(0.2,0.5))
        if self.reading_progress >= 1.0 and self.personality['archetype'] == 'idle_reader': self._transition_to('idling')

    def _state_interacting(self):
        elements = self._find_interactable_elements()
        if elements:
            el = random.choice(elements)
            is_form = el.tag_name.lower() in ['input', 'textarea', 'select']
            if self._human_click(el):
                if is_form and el.tag_name.lower() != 'select' and el.get_attribute('type') not in ['submit','button','checkbox','radio']:
                    self._simulate_typing(el)
                if random.random() < (0.3 - self.cognitive_load*0.2): time.sleep(random.uniform(0.2,0.8)); return
        self._transition_to('browsing')

    def _state_distracted(self):
        for _ in range(random.randint(1,2)):
            act = random.choices(['scroll','pause','tab_switch','wiggle'], weights=[0.3,0.4,0.15,0.15], k=1)[0]
            if act == 'scroll': self._human_scroll(random.uniform(-0.3,0.3))
            elif act == 'pause': time.sleep(random.uniform(1,3))
            elif act == 'tab_switch':
                try: self.driver.switch_to.new_window('tab'); time.sleep(random.uniform(0.3,1)); self.driver.close(); self.driver.switch_to.window(self.driver.window_handles[0])
                except Exception as e: logging.warning(f"Tab switch error: {e}")
            elif act == 'wiggle':
                try:
                    actions = ActionChains(self.driver)
                    for _ in range(random.randint(2,4)): actions.move_by_offset(random.randint(-100,100),random.randint(-100,100)).pause(random.uniform(0.05,0.15))
                    actions.perform()
                except Exception as e: logging.warning(f"Mouse wiggle error: {e}")
            time.sleep(random.uniform(0.2,0.8))
        self._transition_to('browsing')

    def _state_idling(self):
        logging.info(f"Arch: {self.personality['archetype']} idling.")
        if random.random() < 0.1: self._human_scroll(random.uniform(-0.05,0.05))
        elif random.random() < 0.05:
            try: ActionChains(self.driver).move_by_offset(random.randint(-20,20),random.randint(-20,20)).perform()
            except: pass
        self.idle_cycles +=1

    def _state_bouncing(self):
        logging.info(f"Arch: {self.personality['archetype']} bouncing.")
        if random.random() < 0.3 and not any(e['event'] == 'scroll' for e in self.behavior_log):
            self._human_scroll(random.uniform(0.1,0.4) * random.choice([-1,1]))

    def _human_scroll(self, factor):
        try:
            cur_pos = self.driver.execute_script("return window.pageYOffset;")
            page_h = self.driver.execute_script("return document.body.scrollHeight;")
            win_h = self.driver.execute_script("return window.innerHeight;")
            if page_h <= win_h: return
            target_amount = factor * win_h
            target_pos = np.clip(cur_pos + target_amount, 0, page_h - win_h)
            if abs(target_pos-cur_pos) < 5: return
            steps = max(3, int(abs(target_pos-cur_pos) / random.uniform(50,100)))
            path = np.linspace(cur_pos, target_pos, steps, dtype=int)
            for pos in path:
                self.driver.execute_script(f"window.scrollTo(0, {pos});")
                delay = (0.005+random.uniform(0,0.02))/self.personality['behavior']['scroll_speed'] * (1+self.cognitive_load*0.1)
                time.sleep(max(0.001,delay))
            self.behavior_log.append({'time': datetime.now().isoformat(), 'event': 'scroll', 'from_y':int(cur_pos), 'to_y':int(target_pos)})
        except WebDriverException as e: logging.warning(f"Scroll error: {e}")

    def _human_click(self, element):
        tag, text = "unk_tag", "unk_text"
        try:
            if not element: logging.warning("Click on non-existent element."); return False
            try: tag = element.tag_name; text = (element.text[:30]+'...' if len(element.text)>30 else element.text) if element.text else "N/A"
            except StaleElementReferenceException: logging.warning("Element stale for log info.")
            except Exception as e: logging.warning(f"Log info error: {e}")

            self.driver.execute_script("arguments[0].scrollIntoView({behavior:'auto',block:'nearest',inline:'nearest'});", element)
            time.sleep(random.uniform(0.1,0.3)/self.personality['behavior']['scroll_speed'])
            actions = ActionChains(self.driver)
            try:
                WebDriverWait(self.driver,1).until(EC.visibility_of(element))
                size = element.size
                if not(size and size['width']>0 and size['height']>0):
                    self.driver.execute_script("arguments[0].click();", element); return True
                off_x, off_y = int(size['width']*random.uniform(0.2,0.8)), int(size['height']*random.uniform(0.2,0.8))
                actions.move_to_element_with_offset(element,off_x,off_y).pause(random.uniform(0.02,0.1)*(1+self.personality['traits']['motor']['speed_variability'])).click().perform()
                self.behavior_log.append({'time':datetime.now().isoformat(),'event':'click','element_tag':tag,'element_text':text})
                return True
            except (StaleElementReferenceException, TimeoutException, ElementClickInterceptedException, ElementNotInteractableException, MoveTargetOutOfBoundsException) as e:
                logging.warning(f"ActionChains click failed ({type(e).__name__} on {tag}). JS fallback.")
                try: self.driver.execute_script("arguments[0].click();",element); return True
                except Exception as js_e: logging.warning(f"JS fallback fail for {tag}: {js_e}"); return False
            except Exception as e: logging.error(f"Unexpected click error ({type(e).__name__} on {tag}). JS fallback.");
            try: self.driver.execute_script("arguments[0].click();",element); return True
            except Exception as js_e: logging.warning(f"JS fallback (general) fail for {tag}: {js_e}"); return False
        except Exception as e: logging.error(f"Top-level click error for {tag}: {e}", exc_info=False); return False

    def _find_interactable_elements(self, link_only=False):
        base_sel = ["button","input[type='submit']","[role='button']","[onclick]",".btn",".button"]
        link_sel = ["a[href]"]
        content_link_sel = ["article a[href]","main a[href]",".content a[href]",".post a[href]","h1 a[href]","h2 a[href]","h3 a[href]",".nav-link",".menu-item a[href]",".pagination a[href]"]
        sel_to_use = link_sel + content_link_sel
        if not link_only: sel_to_use.extend(base_sel+["input[type='text']","input[type='search']","textarea","select"])
        elements = []
        vp_h, scroll_y = self.driver.execute_script("return window.innerHeight;"), self.driver.execute_script("return window.pageYOffset;")
        for sel in sel_to_use:
            try:
                found = self.driver.find_elements(By.CSS_SELECTOR, sel)
                for el in found:
                    try:
                        if el.is_displayed() and el.is_enabled():
                            size = el.size
                            if not (size and size['width']>5 and size['height']>5): continue
                            if (scroll_y - vp_h*1.0 < el.location['y'] < scroll_y + vp_h*1.5): elements.append(el)
                    except (StaleElementReferenceException,WebDriverException): continue
            except WebDriverException: continue
        return list({el.id:el for el in elements}.values())


    def _random_hover(self):
        elements = self._find_interactable_elements()
        if elements:
            el = random.choice(elements); tag="unk_tag_hover"
            try:
                try: tag = el.tag_name
                except: pass
                WebDriverWait(self.driver,1).until(EC.visibility_of(el))
                actions = ActionChains(self.driver)
                actions.move_to_element_with_offset(el, el.size['width']//2, el.size['height']//2)
                actions.pause(random.uniform(0.1,0.5)*(1+self.personality['traits']['motor']['speed_variability'])).perform()
                self.behavior_log.append({'time':datetime.now().isoformat(),'event':'hover','element_tag':tag})
            except Exception as e: logging.warning(f"Hover error on {tag}: {type(e).__name__}")


    def _simulate_typing(self, element):
        tag="unk_tag_type"
        try:
            try: tag = element.tag_name
            except: pass
            if random.random() < 0.7:
                try: element.clear()
                except ElementNotInteractableException: pass
            text = random.choice(["short","query","test","info",self.personality['archetype'][:5]])
            base_delay = 0.08; mod = (1/max(0.1,self.personality['traits']['motor']['coordination']))*(1+self.cognitive_load*0.3)*(1+self.personality['traits']['motor']['speed_variability']*random.uniform(-0.5,0.5))
            min_d, max_d = 0.02, 0.2
            ActionChains(self.driver).click(element).perform(); time.sleep(random.uniform(0.05,0.15))
            for i,char in enumerate(text):
                delay = np.clip(base_delay*mod*random.uniform(0.7,1.3), min_d, max_d); time.sleep(delay)
                if random.random() < self.personality['behavior']['error_rate']*0.2 and i>0:
                    wrong = random.choice("qwerty")
                    self.driver.execute_script("arguments[0].value+=arguments[1];",element,wrong)
                    time.sleep(np.clip(base_delay*mod*random.uniform(1.2,2),min_d,max_d))
                    self.driver.execute_script("arguments[0].value=arguments[0].value.slice(0,-1);",element)
                    time.sleep(np.clip(base_delay*mod*random.uniform(0.8,1.5),min_d,max_d))
                self.driver.execute_script("arguments[0].value+=arguments[1];",element,char)
            time.sleep(random.uniform(0.1,0.3))
            if (element.get_attribute('type')=='search' or 'search' in element.get_attribute('class') or 'query' in element.get_attribute('name')) and random.random()<0.5:
                element.send_keys(webdriver.common.keys.Keys.ENTER); logging.info("Enter after typing.")
            self.behavior_log.append({'time':datetime.now().isoformat(),'event':'typing','text_typed':text,'element_tag':tag})
        except Exception as e: logging.error(f"Typing error for {tag}: {e}",exc_info=False)

    def _find_and_interact(self, link_only=False):
        elements = self._find_interactable_elements(link_only=link_only)
        if elements:
            target = random.choice(elements); tag,txt="unk_tag_interact","unk_txt_interact"
            try: tag=target.tag_name; txt=(target.text[:20] if target.text else 'N/A')
            except:pass
            current_domain=""
            try: current_domain = self.driver.current_url.split('/')[2]
            except:pass
            internal = [el for el in elements if el.tag_name=='a' and el.get_attribute('href') and current_domain and current_domain in el.get_attribute('href')]
            if (link_only and internal and random.random()<0.8) or (internal and random.random()<0.5): target=random.choice(internal)
            logging.info(f"Interacting with {tag} (Text: {txt})")
            return self._human_click(target)
        logging.info("No interactable elements for interaction."); return False


class NeuroReporter:
    @staticmethod
    def send_report(session_id, personality, behavior_log, session_data):
        subject = f"🧠 #{session_id} | {personality['archetype']} | Ads:{personality['ad_clicks']} | Ref:{personality['fingerprint'].get('referrer_url','Direct') or 'Direct'}"
        duration = session_data['duration']
        min_target_dur = MIN_SESSION_DURATION * personality['behavior']['session_duration_modifier'][0]
        max_target_dur = MAX_SESSION_DURATION * personality['behavior']['session_duration_modifier'][1]

        report_lines = [f"=== NEURO-BOT v{NEURO_VERSION} ===", f"SID: {session_id}",
                        f"Time: {session_data['start_time']} -> {session_data['end_time']}",
                        f"Dur: {duration:.1f}s ({min_target_dur:.0f}-{max_target_dur:.0f}s target)",
                        f"URL: {session_data['url']}", f"Proxy: {session_data.get('proxy_used', 'N/A')}",
                        f"Ref: {personality['fingerprint'].get('referrer_url', 'Direct') or 'Direct'}",
                        f"Arch: {personality['archetype']}, Ads: {personality['ad_clicks']}/{personality['behavior']['max_ad_clicks']}",
                        f"Ad Prob (beh): {personality['behavior']['ad_click_probability']:.2%}",
                        "\n=== ACTIVITY SUMMARY ==="]
        ad_scans = [e for e in behavior_log if e.get('event') == 'ad_scan_completed']
        ad_clicks = [e for e in behavior_log if e.get('event') == 'ad_click']
        report_lines.append(f"Ad Scans: {len(ad_scans)}, Total Ads Visible: {sum(e.get('ads_found_visible',0) for e in ad_scans)}")
        report_lines.append(f"Ads Clicked: {len(ad_clicks)}, Total Actions: {len(behavior_log)}")

        if ad_clicks:
            report_lines.append("\n=== AD CLICK DETAILS (Last 5) ===")
            for click_event in ad_clicks[-5:]:
                pos = click_event.get('ad_position', {'x':'?','y':'?'})
                report_lines.append(f"- Time: {click_event['time']}, Pos: ({pos.get('x','?')},{pos.get('y','?')})")
        else: report_lines.append("\nNo ads clicked.")

        report_lines.extend([
            "\n=== PERSONALITY (Summary) ===",
            f"Cognitive Openness: {personality['traits']['cognitive']['openness']:.2f}, Curiosity: {personality['traits']['cognitive']['curiosity']:.2f}",
            f"Motor Accuracy: {personality['traits']['motor']['accuracy']:.2f}, Speed Var: {personality['traits']['motor']['speed_variability']:.2f}",
            "\n=== BEHAVIOR ===", f"Scroll Factor: {personality['behavior']['scroll_speed']:.2f}x, Read Pattern: {personality['behavior']['reading_pattern']}",
            "\n=== FINGERPRINT ===",
            f"Device: {personality['fingerprint']['device_profile']['type']} (OS: {personality['fingerprint']['device_profile']['os']})",
            f"Screen: {personality['fingerprint']['device_profile']['screen']}, Pixel Ratio: {personality['fingerprint']['device_profile']['pixel_ratio']}",
            "\n=== STATS ===", f"States: {json.dumps(session_data.get('state_counts',{}))}",
            f"Cognitive Load: Avg {session_data.get('avg_cognitive_load',0):.2f}, Max {session_data.get('max_cognitive_load',0):.2f}",
            "\n=== EVENTS (Sample - First 5, Last 5) ==="])

        sample_logs = behavior_log[:5] + ([{'event':'...','time':''}] if len(behavior_log)>10 else []) + (behavior_log[-5:] if len(behavior_log)>5 else behavior_log if len(behavior_log)<=5 else [])
        if 5 < len(behavior_log) <= 10: sample_logs = behavior_log

        for event in sample_logs:
            log_line = f"{event.get('time','')} - {event.get('event','')}"
            details = [f"tag:{event['element_tag']}" if 'element_tag' in event else "",
                       f"text:'{event['element_text']}'" if 'element_text' in event and event['element_text']!="N/A" else "",
                       f"ERROR:{event['error'][:50]}" if 'error' in event else "",
                       f"details:'{str(event['details'])[:50]}'" if 'details' in event else ""]
            details_str = ", ".join(d for d in details if d)
            if details_str: log_line += f" ({details_str})"
            report_lines.append(log_line)

        anomalies = NeuroReporter.detect_anomalies(behavior_log, personality, session_data)
        report_lines.append("\n=== ANOMALIES ===")
        if anomalies: report_lines.extend([f"- {anomaly}" for anomaly in anomalies])
        else: report_lines.append("No significant anomalies.")
        final_report = "\n".join(report_lines)
        logging.info(f"Sending report for SID: {session_id}")
        send_gmail_email_alert(subject=subject, body=final_report, to_email="triggerhappygod@gmail.com")
        report_dir = Path("neuro_reports_illusion_lite"); report_dir.mkdir(exist_ok=True)
        filename = report_dir / f"neuro_report_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        try:
            with open(filename, 'w', encoding='utf-8') as f: f.write(final_report)
            logging.info(f"Report saved: {filename}")
        except IOError as e: logging.error(f"Failed to save report {filename}: {e}")

    @staticmethod
    def detect_anomalies(behavior_log, personality, session_data):
        anomalies = []
        if not behavior_log: return anomalies
        clicks = [datetime.fromisoformat(e['time']) for e in behavior_log if e.get('event') in ['click','ad_click'] and 'time' in e]
        if len(clicks) > 3:
            intervals = [(clicks[i+1]-clicks[i]).total_seconds() for i in range(len(clicks)-1)]
            fast = [i for i in intervals if i<0.3]
            if len(fast) > len(clicks)*0.6 and (sum(fast)/len(fast) if fast else 0) < 0.1:
                anomalies.append(f"High ratio rapid clicks (avg {sum(fast)/len(fast):.2f}s).")
        errors = [e for e in behavior_log if '_error' in e.get('event','').lower() or 'fail' in e.get('event','').lower()]
        total_actions = len(behavior_log)
        if total_actions > 0 and len(errors)/total_actions > 0.35: anomalies.append(f"High error rate: {len(errors)/total_actions:.1%}.")
        scrolls = [e for e in behavior_log if e.get('event')=='scroll']
        if total_actions > 2 and (not scrolls or len(scrolls)/total_actions < 0.01) and personality['archetype']!='bouncer':
            anomalies.append("Low scroll activity for non-bouncer.")

        duration = session_data.get('duration',0)
        min_dur, max_dur = MIN_SESSION_DURATION*personality['behavior']['session_duration_modifier'][0], MAX_SESSION_DURATION*personality['behavior']['session_duration_modifier'][1]
        if personality['archetype']=='bouncer' and duration > max_dur*1.5: anomalies.append(f"Bouncer session too long: {duration:.1f}s (exp <{max_dur:.1f}s).")
        elif personality['archetype']!='bouncer' and duration < min_dur*0.5: anomalies.append(f"Session too short for {personality['archetype']}: {duration:.1f}s (exp >{min_dur:.1f}s).")
        if duration > 0 and total_actions/duration > 10: anomalies.append(f"High action freq: {total_actions/duration:.1f} actions/sec.")
        if personality['ad_clicks'] > personality['behavior']['max_ad_clicks']: anomalies.append(f"Exceeded max ad clicks ({personality['ad_clicks']}/{personality['behavior']['max_ad_clicks']}).")
        return anomalies


class NeuroProxyManager:
    @staticmethod
    def validate_proxy(proxy):
        if not proxy or ':' not in proxy: logging.warning(f"Invalid proxy: {proxy}"); return False
        fmt_proxy = proxy if proxy.startswith("http") else f"http://{proxy}"
        proxies = {"http":fmt_proxy, "https":fmt_proxy}
        for url in PROXY_TEST_URLS:
            try:
                r = requests.get(url,proxies=proxies,timeout=PROXY_TEST_TIMEOUT,verify=False,allow_redirects=True,
                                 headers={'User-Agent':UserAgent(fallback="Mozilla/5.0").random})
                if r.status_code==200: logging.info(f"Proxy {proxy} OK with {url}"); return True
                logging.warning(f"Proxy {proxy} fail {url} (Status:{r.status_code}).")
            except requests.exceptions.RequestException as e: logging.warning(f"Proxy {proxy} fail {url} (Err:{type(e).__name__}).")
            except Exception as e: logging.error(f"Proxy {proxy} validation {url} generic error: {e}", exc_info=False)
        logging.warning(f"Proxy {proxy} failed all tests."); return False

    @staticmethod
    def get_proxy(available_proxies):
        if not available_proxies: logging.warning("No proxies."); return None
        shuffled = random.sample(available_proxies, len(available_proxies))
        for i in range(min(PROXY_MAX_ATTEMPTS, len(shuffled))):
            p = shuffled[i]
            logging.info(f"Validating proxy: {p} (Attempt {i+1})")
            if NeuroProxyManager.validate_proxy(p): logging.info(f"Validated: {p}"); return p
            logging.warning(f"Proxy {p} failed validation."); time.sleep(0.5)
        logging.error(f"Failed all {PROXY_MAX_ATTEMPTS} proxy attempts."); return None


def visit_blog(session_id, target_url, proxy=None):
    start_time_obj = datetime.now()
    personality = NeuroPersonalityCore.generate_personality()
    device_profile = personality['fingerprint']['device_profile']
    chosen_referrer = personality['fingerprint'].get('referrer_url')

    logging.info(f"NeuroAgent #{session_id} STARTING. Arch: {personality['archetype']}, Dev: {device_profile['type']}, Proxy: {proxy or 'Direct'}, Ref: {chosen_referrer or 'Direct'}")

    driver = None
    autopilot = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            logging.info(f"NeuroAgent #{session_id} - Browser launch attempt {attempt + 1}/{MAX_RETRIES + 1}")
            options = Options()
            options.add_argument("--disable-blink-features=AutomationControlled")
            options.add_argument("--disable-extensions")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-gpu")
            options.add_argument("--disable-infobars")
            options.add_argument("--disable-popup-blocking")
            options.add_argument('--ignore-certificate-errors')
            options.add_argument('--ignore-ssl-errors')

            ua_gen = UserAgent(fallback="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36")
            ua_str = ""
            width, height = 0, 0 # Initialize
            if device_profile['type'] == 'mobile':
                ua_str = ua_gen.android if 'Android' in device_profile['os'] else ua_gen.iphone
                width, height = map(int, device_profile['screen'].split('x'))
                mobile_emulation = {"deviceMetrics": {"width":width,"height":height,"pixelRatio":device_profile['pixel_ratio']}, "userAgent":ua_str}
                options.add_experimental_option("mobileEmulation", mobile_emulation)
            elif device_profile['type'] == 'tablet':
                ua_str = ua_gen.android if 'Android' in device_profile['os'] else ua_gen.ipad
                width, height = map(int, device_profile['screen'].split('x'))
                options.add_argument(f"--window-size={width},{height}")
                options.add_argument(f"user-agent={ua_str}")
            else: # Desktop
                ua_str = ua_gen.chrome
                width, height = map(int, device_profile['screen'].split('x'))
                options.add_argument(f"--window-size={width},{height}")
                options.add_argument(f"user-agent={ua_str}")

            options.add_experimental_option("excludeSwitches", ["enable-automation", "enable-logging"])
            options.add_experimental_option('useAutomationExtension', False)
            if platform.system()=="Linux": options.add_argument('--disable-setuid-sandbox')
            if proxy: options.add_argument(f"--proxy-server={proxy if proxy.startswith('http') else 'http://'+proxy}")

            driver = webdriver.Chrome(options=options)
            driver.set_page_load_timeout(45)
            driver.set_script_timeout(25)

            stealth="Object.defineProperty(navigator,'webdriver',{get:()=>undefined});Object.defineProperty(navigator,'languages',{get:()=>['en-US','en']});Object.defineProperty(navigator,'plugins',{get:()=>[1,2,3]});"
            driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {"source":stealth})

            if chosen_referrer:
                try:
                    logging.info(f"NeuroAgent #{session_id} setting Referer via CDP: {chosen_referrer}")
                    driver.execute_cdp_cmd('Network.enable',{})
                    driver.execute_cdp_cmd('Network.setExtraHTTPHeaders',{'headers':{'Referer':chosen_referrer}})
                except Exception as e: logging.warning(f"NeuroAgent #{session_id} CDP Referer fail: {e}")

            logging.info(f"NeuroAgent #{session_id} navigating to: {target_url}")
            driver.get(target_url)

            logging.info(f"NeuroAgent #{session_id} handling consent...")
            consent_clicked, consent_wait = False, 7
            common_texts = ["consent","accept all","accept","agree","i agree","ok","got it","allow all","allow cookies"]
            xpaths = [f"//button[normalize-space(translate(.,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'))='{t}']|//a[normalize-space(translate(.,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'))='{t}']|//div[@role='button' and normalize-space(translate(.,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'))='{t}']" for t in common_texts]
            xpaths.extend([f"//button[contains(translate(normalize-space(.),'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'{t}')]|//a[contains(translate(normalize-space(.),'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'{t}')]" for t in common_texts])

            init_consent_time = time.time()
            def click_consent(s_driver, xpaths_list):
                for part in xpaths_list:
                    for xp in part.split(" | "):
                        try:
                            btn=WebDriverWait(s_driver,0.2).until(EC.element_to_be_clickable((By.XPATH,xp)))
                            s_driver.execute_script("arguments[0].click();",btn); logging.info(f"Clicked consent via JS: {xp[:50]}..."); return True
                        except: continue
                return False

            if click_consent(driver, xpaths): consent_clicked = True
            if not consent_clicked and time.time()-init_consent_time < consent_wait/2 :
                iframes = driver.find_elements(By.TAG_NAME,"iframe")
                for iframe in iframes:
                    if consent_clicked: break
                    try:
                        id_ = iframe.get_attribute("id") or iframe.get_attribute("name") or "unk_iframe"
                        driver.switch_to.frame(iframe)
                        if click_consent(driver,xpaths): consent_clicked=True
                        driver.switch_to.default_content()
                    except Exception as e: logging.warning(f"Iframe {id_} error: {e}"); driver.switch_to.default_content()

            logging.info(f"NeuroAgent #{session_id} consent: {'Clicked' if consent_clicked else 'None/Timeout'}")
            if consent_clicked: time.sleep(random.uniform(0.5,1.5))

            autopilot = AutoPilot(driver, personality)
            min_dur = MIN_SESSION_DURATION * personality['behavior']['session_duration_modifier'][0]
            max_dur = MAX_SESSION_DURATION * personality['behavior']['session_duration_modifier'][1]
            duration_s = random.uniform(min_dur, max_dur)
            if personality['archetype']=='bouncer': duration_s = random.uniform(5,15)
            session_end = time.time() + duration_s
            logging.info(f"NeuroAgent #{session_id} session start. Est Dur: {duration_s:.1f}s. Arch: {personality['archetype']}")

            while time.time() < session_end:
                autopilot.step()
                time.sleep(autopilot.get_behavioral_delay())
                if len([e for e in autopilot.behavior_log if "_error" in e.get('event','').lower()]) > 5:
                    logging.warning(f"NeuroAgent #{session_id} error limit. Ending early."); break
                if personality['archetype']=='bouncer' and (time.time()-start_time_obj.timestamp()) > duration_s:
                    logging.info(f"Bouncer {session_id} short limit."); break

            actual_dur = (datetime.now()-start_time_obj).total_seconds()
            logging.info(f"NeuroAgent #{session_id} session end. Actual Dur: {actual_dur:.1f}s")

            session_data = {
                'start_time':start_time_obj.isoformat(), 'end_time':datetime.now().isoformat(),
                'duration':actual_dur, 'url':target_url, 'proxy_used':proxy or "Direct", 'state_counts':{},
                'avg_cognitive_load':0,'max_cognitive_load':0,
            }
            states = [log.get('to',log.get('state')) for log in autopilot.behavior_log if log.get('event')=='state_change' or 'state' in log]
            session_data['state_counts'] = {s:states.count(s) for s in set(states) if s}
            loads = [log['cognitive_load'] for log in autopilot.behavior_log if 'cognitive_load' in log and isinstance(log['cognitive_load'],(int,float))]
            if loads: session_data['avg_cognitive_load']=sum(loads)/len(loads); session_data['max_cognitive_load']=max(loads)
            NeuroReporter.send_report(session_id,personality,autopilot.behavior_log,session_data)
            logging.info(f"NeuroAgent #{session_id} FINISHED successfully.")
            return

        except WebDriverException as e: logging.error(f"NeuroAgent #{session_id} WebDriverErr (Atmpt {attempt+1}): {type(e).__name__} - {str(e)[:200]}", exc_info=False)
        except Exception as e:
            is_autopilot_err = autopilot and hasattr(autopilot,'behavior_log') and autopilot.behavior_log and \
                               any(log.get('event')=='autopilot_step_error' and log.get('details')==str(e) for log in autopilot.behavior_log)
            if is_autopilot_err: logging.error(f"NeuroAgent #{session_id} SessionErr(AutoPilot) (Atmpt {attempt+1}): {type(e).__name__} - {e}", exc_info=False)
            else: logging.error(f"NeuroAgent #{session_id} GeneralErr (Atmpt {attempt+1}): {type(e).__name__} - {e}", exc_info=False)
        finally:
            if driver:
                try: driver.quit(); logging.info(f"NeuroAgent #{session_id} browser closed (Atmpt {attempt+1}).")
                except Exception as e_q: logging.warning(f"Driver quit err {session_id} (Atmpt {attempt+1}): {e_q}")
        if attempt < MAX_RETRIES:
            delay = RETRY_DELAY_BASE*(1.5**attempt) + random.uniform(0,RETRY_DELAY_BASE/2)
            logging.info(f"Retrying NeuroAgent #{session_id} in {delay:.1f}s...")
            time.sleep(delay)

    logging.critical(f"NeuroAgent #{session_id} FAILED ALL {MAX_RETRIES+1} ATTEMPTS. Proxy: {proxy}. URL: {target_url}. Arch: {personality.get('archetype','N/A') if 'personality' in locals() else 'N/A'}")


class NeuroThreadManager:
    def __init__(self, target_urls):
        self.session_counter = 1
        self.active_threads = []
        self.trusted_proxies = []
        self.last_proxy_refresh = 0
        self.proxy_failures = {}
        self.target_urls = target_urls if isinstance(target_urls, list) else [target_urls]
        self._load_trusted_proxies()

        self.daily_session_count = 0
        self.last_daily_reset_date = date.today() - timedelta(days=1) # Force initial update
        self.current_max_threads = MAX_THREADS_DEFAULT
        self._update_overmind_params() # Initialize DAILY_VISIT_QUOTA via global modification
        self.initial_daily_quota = DAILY_VISIT_QUOTA

        self.manager_state = 'spawning'
        self.current_spawning_start_time = time.time()
        self.active_spawning_duration_max = 10 * 60 # 10 mins
        self.min_cooldown_seconds = 15 * 60 # 15 mins
        self.max_cooldown_seconds = 25 * 60 # 25 mins (avg 20 min cooldown)

    def _update_overmind_params(self):
        global DAILY_VISIT_QUOTA # This method modifies the global DAILY_VISIT_QUOTA
        if date.today() > self.last_daily_reset_date:
            new_daily_quota = random.randint(500, 1000) # Target daily views
            logging.info(f"Daily reset: Count {self.daily_session_count} -> 0. Old Quota: {DAILY_VISIT_QUOTA}, New Quota: {new_daily_quota}")
            self.daily_session_count = 0
            self.last_daily_reset_date = date.today()
            DAILY_VISIT_QUOTA = new_daily_quota
            self.initial_daily_quota = new_daily_quota

        hour = datetime.now().hour
        if 7<=hour<=10 or 13<=hour<=16: self.current_max_threads = int(MAX_THREADS_DEFAULT*random.uniform(1.0,1.3))
        elif 19<=hour<=22: self.current_max_threads = int(MAX_THREADS_DEFAULT*random.uniform(0.9,1.2))
        elif 0<=hour<=5: self.current_max_threads = int(MAX_THREADS_DEFAULT*random.uniform(0.5,0.7))
        else: self.current_max_threads = MAX_THREADS_DEFAULT
        self.current_max_threads = max(1,self.current_max_threads)

    def _load_trusted_proxies(self):
        self.trusted_proxies = list(TRUSTED_PROXIES)
        env_proxies = os.environ.get("HTTP_PROXIES_LIST")
        if env_proxies:
            loaded = [p.strip() for p in env_proxies.split(',') if p.strip()]
            self.trusted_proxies.extend(loaded); logging.info(f"Loaded {len(loaded)} proxies from env.")
        self.trusted_proxies = list(set(self.trusted_proxies))
        logging.info(f"Total unique trusted proxies: {len(self.trusted_proxies)}")
        if not self.trusted_proxies: logging.warning("No trusted proxies.")

    def refresh_proxies(self):
        if time.time() - self.last_proxy_refresh > PROXY_REFRESH_INTERVAL:
            logging.info("Refreshing proxy list evaluation.")
            self.proxy_failures = {p:c//2 for p,c in self.proxy_failures.items() if c>0}
            self.last_proxy_refresh = time.time()

    def cleanup_threads(self):
        self.active_threads = [t for t in self.active_threads if t.is_alive()]

    def run_session(self):
        if self.daily_session_count >= DAILY_VISIT_QUOTA:
            logging.info(f"Quota hit ({self.daily_session_count}/{DAILY_VISIT_QUOTA}) in run_session. No spawn."); return False
        self.refresh_proxies()
        proxy = None
        avail_proxies = [p for p in self.trusted_proxies if self.proxy_failures.get(p,0)<3]
        if random.random()<0.85 and avail_proxies:
            selected = NeuroProxyManager.get_proxy(avail_proxies)
            if selected: proxy = selected
            else: logging.warning(f"SID {self.session_counter}: No working proxy found. Direct.")

        url = random.choice(self.target_urls)
        thread = threading.Thread(target=visit_blog,args=(self.session_counter,url,proxy),daemon=True)
        thread.name = f"NeuroAgent-{self.session_counter}"
        thread.start()
        self.active_threads.append(thread)
        self.daily_session_count += 1
        logging.info(f"Launch SID {self.session_counter} for {url} (Proxy:{proxy or 'Direct'}). Threads:{len(self.active_threads)}. Daily:{self.daily_session_count}/{DAILY_VISIT_QUOTA}. MaxThrds:{self.current_max_threads}")
        self.session_counter += 1
        
        avg_sess_est = (MIN_SESSION_DURATION+MAX_SESSION_DURATION)/2
        ideal_stagger = max(1.0, avg_sess_est/max(1,self.current_max_threads)*0.5)
        stagger = np.clip(random.uniform(ideal_stagger*0.7,ideal_stagger*1.3),1,60)
        logging.debug(f"Stagger time: {stagger:.1f}s")
        time.sleep(stagger)
        return True

    def run(self):
        logging.info(f"=== NeuroBot v{NEURO_VERSION} Start (Illusion Lite) ===")
        logging.info(f"=== URLs: {self.target_urls} ===")
        logging.info(f"=== Initial Daily Quota: {self.initial_daily_quota} (Target: {DAILY_VISIT_QUOTA} daily) ===")
        logging.info(f"=== Base Max Threads: {MAX_THREADS_DEFAULT} ===")
        logging.info(f"=== Pulsing: Spawn ~{self.active_spawning_duration_max/60:.0f}m, Cooldown ~{self.min_cooldown_seconds/60:.0f}-{self.max_cooldown_seconds/60:.0f}m ===")

        if not self.target_urls or not any(url.strip() for url in self.target_urls if isinstance(url,str)):
            logging.critical("No valid target URLs. Exiting."); return

        try:
            while True:
                self._update_overmind_params(); self.cleanup_threads()

                if self.daily_session_count >= DAILY_VISIT_QUOTA:
                    logging.info(f"Daily quota ({self.daily_session_count}/{DAILY_VISIT_QUOTA}) met. Cooldown until next day.")
                    self.manager_state = 'cooldown'
                    now = datetime.now(); tomorrow = datetime.combine(now.date()+timedelta(days=1),datetime.min.time())
                    sleep_dur = (tomorrow-now).total_seconds() + random.uniform(60,300)
                    logging.info(f"Sleeping ~{sleep_dur/3600:.2f} hrs.")
                    time.sleep(sleep_dur)
                    self._update_overmind_params() # Force daily reset
                    self.manager_state = 'spawning'; self.current_spawning_start_time = time.time()
                    logging.info("Daily quota reset. Resuming spawning."); continue

                if self.manager_state == 'spawning':
                    if (time.time()-self.current_spawning_start_time) > self.active_spawning_duration_max:
                        logging.info(f"Spawn window ({self.active_spawning_duration_max/60:.0f}m) ended. Cooldown. Daily:{self.daily_session_count}/{DAILY_VISIT_QUOTA}")
                        self.manager_state = 'cooldown'
                    else:
                        if len(self.active_threads) < self.current_max_threads and self.daily_session_count < DAILY_VISIT_QUOTA:
                            logging.debug(f"Spawning. Active:{len(self.active_threads)}, Max:{self.current_max_threads}, Daily:{self.daily_session_count}/{DAILY_VISIT_QUOTA}")
                            if not self.run_session() and self.daily_session_count >= DAILY_VISIT_QUOTA: # Quota might be hit by run_session itself
                                logging.info("Quota hit by run_session. Cooldown."); self.manager_state = 'cooldown'
                        elif len(self.active_threads) >= self.current_max_threads:
                            rem_spawn_t = self.active_spawning_duration_max - (time.time()-self.current_spawning_start_time)
                            logging.debug(f"Max threads ({self.current_max_threads}) in spawn. Waiting. ({rem_spawn_t/60:.1f}m left)")
                            time.sleep(random.uniform(3,7))
                        # Implicit else: quota is hit (self.daily_session_count >= DAILY_VISIT_QUOTA), will be handled by top block
                
                elif self.manager_state == 'cooldown':
                    if self.daily_session_count < DAILY_VISIT_QUOTA: # Cooldown only if quota not met
                        cd_dur = random.uniform(self.min_cooldown_seconds, self.max_cooldown_seconds)
                        logging.info(f"Cooldown. Daily:{self.daily_session_count}/{DAILY_VISIT_QUOTA}. Sleep {cd_dur/60:.1f}m.")
                        time.sleep(cd_dur)
                        logging.info("Cooldown end. Spawning.")
                        self.manager_state='spawning'; self.current_spawning_start_time=time.time()
                    else: # Quota met during cooldown phase itself or spawning. Defer to top block.
                        logging.debug(f"Cooldown but quota met ({DAILY_VISIT_QUOTA}). Main quota logic will take over.")
                        time.sleep(60)

                # Short general sleep to prevent tight loops if no other sleep condition met
                if not (self.manager_state=='spawning' and len(self.active_threads)<self.current_max_threads and self.daily_session_count < DAILY_VISIT_QUOTA):
                    time.sleep(random.uniform(1,3))

        except KeyboardInterrupt: logging.info("KeyboardInterrupt. Shutting down...")
        except Exception as e: logging.critical(f"Manager CRITICAL ERROR: {e}",exc_info=True)
        finally:
            logging.info("Manager shutdown. Waiting for agent threads...")
            for t in self.active_threads:
                if t.is_alive(): logging.info(f"Wait for {t.name} (10s)..."); t.join(timeout=10)
            logging.info("All agent threads handled. Exiting manager.")


# --- Email Configuration & Firebase ---
def send_gmail_email_alert(subject, body, to_email):
    from_email = os.getenv("FIREBASE_GMAIL_USER")
    from_password = os.getenv("FIREBASE_GMAIL_APP_PASSWORD")
    if not from_email or not from_password: return
    msg = MIMEMultipart(); msg['From'] = from_email; msg['To'] = to_email; msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain', _charset='utf-8'))
    for _ in range(2):
        try:
            with smtplib.SMTP('smtp.gmail.com', 587) as server:
                server.starttls(); server.login(from_email, from_password); server.send_message(msg)
                logging.info(f"✅ Alert '{subject[:30]}...' sent to {to_email}"); return
        except Exception as e: logging.warning(f"❌ Alert send fail: {e}"); time.sleep(3)

FIREBASE_SERVICE_ACCOUNT_PATH = r"C:\Users\Sean\Documents\agenets-workers-to-send-emails-firebase-adminsdk-fbsvc-00510b8385.json" # Path or None
GMAIL_USER_QUEUE = "triggerhappygod@gmail.com"
GMAIL_PASS_QUEUE = "ebmmdrauvfpderdr" # App Password
db = None
USE_FIREBASE_QUEUE = False

if USE_FIREBASE_QUEUE:
    try:
        if FIREBASE_SERVICE_ACCOUNT_PATH and Path(FIREBASE_SERVICE_ACCOUNT_PATH).is_file():
            if not firebase_admin._apps:
                cred = credentials.Certificate(FIREBASE_SERVICE_ACCOUNT_PATH)
                firebase_admin.initialize_app(cred); logging.info("Firebase SDK init OK.")
            db = firestore.client()
        else: logging.warning("Firebase Svc Acc path invalid/missing. FB Queue disabled."); USE_FIREBASE_QUEUE=False
    except Exception as e: logging.critical(f"Firebase init fail: {e}"); db=None; USE_FIREBASE_QUEUE=False
else: logging.info("Firebase queue disabled by USE_FIREBASE_QUEUE setting.")


def send_queued_email(to, subject, body, html_body=None):
    if not (GMAIL_USER_QUEUE and GMAIL_PASS_QUEUE): logging.warning("FB GMAIL creds not set."); return False
    msg = MIMEText(html_body or body, "html" if html_body else "plain", _charset='utf-8')
    msg["Subject"]=subject; msg["From"]=GMAIL_USER_QUEUE; msg["To"]=to
    for _ in range(2):
        try:
            with smtplib.SMTP("smtp.gmail.com",587) as s:
                s.starttls();s.login(GMAIL_USER_QUEUE,GMAIL_PASS_QUEUE);s.send_message(msg)
                logging.info(f"✅ Queued email '{subject[:30]}...' sent to {to}"); return True
        except Exception as e: logging.warning(f"❌ Queued email send fail: {e}"); time.sleep(3)
    logging.error(f"❌ Failed queued email to {to}."); return False

def process_email_queue():
    if db is None or not USE_FIREBASE_QUEUE: logging.info("FB Queue inactive. Processor not starting."); return
    logging.info("FB Email Queue Processor Start.")
    while True:
        try:
            docs = db.collection("mail_queue").limit(5).stream(); processed=0
            for doc in docs:
                data = doc.to_dict()
                to, subj = data.get("to"), data.get("message",{}).get("subject","No Subject")
                body, html = data.get("message",{}).get("text",""), data.get("message",{}).get("html")
                if to and subj and (body or html):
                    if send_queued_email(to,subj,body,html): doc.reference.delete(); processed+=1
                    else: logging.error(f"Failed FB email job {doc.id}, will retry.")
                else: logging.warning(f"Invalid FB email job {doc.id}, deleting."); doc.reference.delete()
            if processed==0: logging.debug("No new FB emails this check.")
            elif processed > 0: logging.info(f"Processed {processed} FB emails.")
        except Exception as e: logging.critical(f"FB Email Queue CRITICAL: {e}",exc_info=True)
        finally: time.sleep(random.uniform(25,45))


# --- Main Execution ---
if __name__ == "__main__":
    target_site_list = [
        "https://thedealsdetective.blogspot.com/",
    ]
    if not any(url.strip() for url in target_site_list if isinstance(url, str)):
        logging.critical("No valid target URLs. Exiting."); sys.exit(1)

    # os.environ["FIREBASE_GMAIL_USER"] = "your_alert_email@gmail.com"
    # os.environ["FIREBASE_GMAIL_APP_PASSWORD"] = "your_gmail_app_password"

    if USE_FIREBASE_QUEUE and db is not None:
        email_thread = threading.Thread(target=process_email_queue, daemon=True, name="EmailQueueProcessor")
        email_thread.start(); logging.info("Firebase Email Queue Processor thread started.")
    else:
        logging.info("Firebase Email Queue Processor NOT started (FB disabled or DB init fail).")

    manager = NeuroThreadManager(target_urls=target_site_list)
    manager.run()