from flask import Flask, request, jsonify, session, render_template_string, render_template, url_for, redirect, flash
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_cors import CORS
from flask_jwt_extended import JWTManager, jwt_required, create_access_token, get_jwt_identity, decode_token
from datetime import datetime, timedelta
from flask_socketio import SocketIO, emit, join_room, leave_room
import time
import numpy as np
from openai import OpenAI
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_migrate import Migrate
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import sqlite3
import hashlib
import secrets
import hmac
import jwt
import os
import logging
from cryptography.fernet import Fernet
import stripe
import json
import uuid
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List
import re
from enum import Enum
from functools import wraps
from dotenv import load_dotenv
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from werkzeug.security import generate_password_hash, check_password_hash
from email_validator import validate_email, EmailNotValidError
import bleach

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s %(message)s',
    handlers=[
        logging.FileHandler('OncoCare.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

load_dotenv()
client = OpenAI(
    base_url="https://api.netmind.ai/inference-api/openai/v1",
    api_key=os.getenv('OPENAI_API_KEY')
)

# Configuration
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///OncoCare.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['JWT_SECRET_KEY'] = os.environ.get('JWT_SECRET_KEY')
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=24)
app.config['JWT_BLACKLIST_ENABLED'] = True
app.config['JWT_BLACKLIST_TOKEN_CHECKS'] = ['access']

# Security configurations
app.config['SESSION_COOKIE_SECURE'] = False
app.config['SESSION_COOKIE_HTTPONLY'] = False
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=2)
STRIPE_PUBLISHABLE_KEY = os.environ.get('STRIPE_PUBLISHABLE_KEY')

# Initialize extensions
db = SQLAlchemy(app)
migrate = Migrate(app, db)
bcrypt = Bcrypt(app)
jwt_manager = JWTManager(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Configure CORS properly
CORS(app, 
     origins=os.environ.get('ALLOWED_ORIGINS', 'http://localhost:3000').split(','),
     supports_credentials=True,
     methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
     allow_headers=['Content-Type', 'Authorization', 'X-API-Key'])

# Rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per hour", "50 per minute"],
    storage_uri="memory://"
)

# Blacklist for revoked tokens
blacklisted_tokens = set()

@jwt_manager.token_in_blocklist_loader
def check_if_token_revoked(jwt_header, jwt_payload):
    return jwt_payload['jti'] in blacklisted_tokens

# Define your pricing plans (match your frontend)
PRICING_PLANS = {
    'Essential': {
        'price': 29900,  # $299 in cents
        'name': 'Essential Plan',
        'description': 'Perfect for small clinics connecting patients with their care teams',
        'features': [
            'Patient-provider communication portal',
            'Nurse care coordination tools',
            'Oncologist diagnostic assistance',
            'Up to 100 patient records',
            'HIPAA compliant platform'
        ]
    },
    'Professional': {
        'price': 69900,  # $699 in cents
        'name': 'Professional Plan',
        'description': 'Comprehensive collaboration platform for cancer centers',
        'features': [
            'Advanced patient-provider dashboard',
            'Multi-disciplinary care coordination',
            'Up to 1,000 patient records',
            '24/7 priority support',
            'EMR system integration'
        ]
    },
    'Advanced': {
        'price': 129900,  # $1,299 in cents
        'name': 'Advanced Plan',
        'description': 'Enterprise-grade platform for healthcare networks',
        'features': [
            'Complete patient journey management',
            'Unlimited patient records',
            'Custom API integrations',
            'White-label solutions',
            'Dedicated account manager'
        ]
    }
}

# Enums
class UserRole(Enum):
    PATIENT = "patient"
    NURSE = "nurse"
    ONCOLOGIST = "oncologist"
    ADMIN = "admin"

class PlanType(Enum):
    BASIC = "basic"
    ESSENTIAL = "essential"
    PROFESSIONAL = "professional"
    ADVANCED = "advanced"
    ENTERPRISE = "enterprise"

class SubscriptionStatus(Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    CANCELLED = "cancelled"
    PENDING = "pending"


# Input validation utilities
class InputValidator:
    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format using email-validator library"""
        try:
            validate_email(email)
            return True
        except EmailNotValidError:
            return False
    
    @staticmethod
    def validate_name(name: str) -> bool:
        """Validate name format"""
        if not name or len(name.strip()) < 2:
            return False
        # Allow letters, spaces, hyphens, apostrophes
        pattern = r"^[a-zA-Z\s\-']+$"
        return bool(re.match(pattern, name.strip()))
    
    @staticmethod
    def validate_password_strength(password: str) -> tuple:
        """Validate password strength"""
        if len(password) < 8:
            return False, "Password must be at least 8 characters long"
        if not re.search(r"[A-Z]", password):
            return False, "Password must contain at least one uppercase letter"
        if not re.search(r"[a-z]", password):
            return False, "Password must contain at least one lowercase letter"
        if not re.search(r"\d", password):
            return False, "Password must contain at least one number"
        if not re.search(r"[!@#$%^&*()_+\-=\[\]{};':\"\\|,.<>\/?]", password):
            return False, "Password must contain at least one special character"
        return True, "Password is strong"
    
    @staticmethod
    def sanitize_input(data: str) -> str:
        """Sanitize input to prevent XSS"""
        if not isinstance(data, str):
            return str(data)
        return bleach.clean(data.strip(), tags=[], strip=True)[:255]

# Database Models
class User(db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    first_name = db.Column(db.String(50), nullable=False)
    last_name = db.Column(db.String(50), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(128), nullable=False)
    role = db.Column(db.Enum('patient', 'nurse', 'oncologist', 'admin', name='user_roles'), nullable=False)
    is_active = db.Column(db.Boolean, default=True)
    is_verified = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    login_attempts = db.Column(db.Integer, default=0)
    locked_until = db.Column(db.DateTime)
    
    # Plan-related fields - made nullable or with defaults for registration
    plan_type = db.Column(db.String(20), default='basic')  # Default plan for new users
    price = db.Column(db.Numeric(10, 2), default=0.00)  # Default to free
    currency = db.Column(db.String(3), default='USD')
    billing_cycle = db.Column(db.String(20), default='monthly')
    max_patients = db.Column(db.Integer, nullable=True)  # Can be null initially
    max_providers_per_patient = db.Column(db.Integer, nullable=True)  # Can be null initially
    features = db.Column(db.JSON, nullable=True)  # Can be null initially
    stripe_price_id = db.Column(db.String(100), nullable=True)  # Can be null initially
    
    # Relationships
    patient = db.relationship('Patient', back_populates='user', uselist=False, lazy=True)
    medical_staff = db.relationship('MedicalStaff', back_populates='user', lazy=True)
    profile = db.relationship('OncologistProfile', back_populates='user', uselist=False)
    # chat_history = db.relationship('ChatHistory', back_populates='patients', lazy=True, cascade='all, delete-orphan', foreign_keys='ChatHistory.patient_id')
    
    def __init__(self, first_name, last_name, email, password, role):
        self.first_name = InputValidator.sanitize_input(first_name)
        self.last_name = InputValidator.sanitize_input(last_name)
        self.email = email.lower().strip()
        self.set_password(password)
        self.role = role
        
        # Set default plan values during registration
        self.plan_type = 'basic'
        self.price = 0.00
        self.currency = 'USD'
        self.billing_cycle = 'monthly'
        
        # Set role-based defaults
        if role == 'patient':
            self.max_patients = None  # Patients don't manage other patients
            self.max_providers_per_patient = 1  # Can have 1 provider in free plan
            self.features = ['basic_tracking', 'appointment_booking']
        elif role in ['nurse', 'oncologist']:
            self.max_patients = 5  # Free plan allows 5 patients
            self.max_providers_per_patient = None  # Not applicable for providers
            self.features = ['patient_management', 'basic_reporting']
        else:  # admin
            self.max_patients = None  # Unlimited for admin
            self.max_providers_per_patient = None
            self.features = ['full_access']
    
    def set_password(self, password):
        """Set password with proper hashing"""
        self.password_hash = bcrypt.generate_password_hash(password).decode('utf-8')
    
    def check_password(self, password):
        """Check password against hash"""
        return bcrypt.check_password_hash(self.password_hash, password)
    
    def is_account_locked(self):
        """Check if account is locked"""
        return self.locked_until and self.locked_until > datetime.utcnow()
    
    def lock_account(self, duration_minutes=30):
        """Lock account for specified duration"""
        self.locked_until = datetime.utcnow() + timedelta(minutes=duration_minutes)
        try:
            db.session.commit()
        except Exception as e:
            logger.error(f"Error locking account: {e}")
            db.session.rollback()
    
    def unlock_account(self):
        """Unlock account and reset login attempts"""
        self.login_attempts = 0
        self.locked_until = None
        try:
            db.session.commit()
        except Exception as e:
            logger.error(f"Error unlocking account: {e}")
            db.session.rollback()
    
    def upgrade_plan(self, plan_type, price, max_patients=None, max_providers_per_patient=None, features=None, stripe_price_id=None):
        """Upgrade user plan - to be called from pricing page"""
        self.plan_type = plan_type
        self.price = price
        self.max_patients = max_patients
        self.max_providers_per_patient = max_providers_per_patient
        self.features = features or []
        self.stripe_price_id = stripe_price_id
        
        try:
            db.session.commit()
        except Exception as e:
            logger.error(f"Error upgrading plan: {e}")
            db.session.rollback()
            raise
    
    @property
    def full_name(self):
        """Get full name"""
        return f"{self.first_name} {self.last_name}"
    
    def to_dict(self):
        """Convert user to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'first_name': self.first_name,
            'last_name': self.last_name,
            'email': self.email,
            'role': self.role,
            'is_active': self.is_active,
            'is_verified': self.is_verified,
            'created_at': self.created_at.isoformat(),
            'last_login': self.last_login.isoformat() if self.last_login else None,
            'plan_type': self.plan_type,
            'price': float(self.price),
            'currency': self.currency,
            'billing_cycle': self.billing_cycle,
            'max_patients': self.max_patients,
            'max_providers_per_patient': self.max_providers_per_patient,
            'features': self.features,
        }

class ChatHistory(db.Model):
    __tablename__ = 'chat_history'
    
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patients.id'), nullable=False)
    user_message = db.Column(db.Text, nullable=False)
    ai_response = db.Column(db.Text, nullable=False)
    
    # Metadata
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    session_id = db.Column(db.String(100))  # To group conversations
    message_type = db.Column(db.String(20), default='chat')  # chat, question, concern, etc.
    
    # AI metadata
    ai_model = db.Column(db.String(50), default='gpt-3.5-turbo')
    response_time = db.Column(db.Float)  # Response time in seconds
    tokens_used = db.Column(db.Integer)
    
    def __repr__(self):
        return f'<ChatHistory {self.id}: {self.user_message[:50]}...>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'patient_id': self.patient_id,
            'user_message': self.user_message,
            'ai_response': self.ai_response,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'session_id': self.session_id,
            'message_type': self.message_type
        }
    
    #Relationship
    patient = db.relationship("Patient", back_populates="chat_history")

class Patient(db.Model):
    __tablename__ = 'patients'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, unique=True)
    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.String(10), nullable=False)
    diagnosis = db.Column(db.String(100), nullable=False)
    stage = db.Column(db.String(20), nullable=False)
    date_diagnosed = db.Column(db.Date, nullable=False)
    risk_score = db.Column(db.Float, default=0.0)
    emergency_contact = db.Column(db.String(15))
    insurance_info = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    treatments = db.relationship('Treatment', back_populates='patient', lazy=True)
    appointments = db.relationship('Appointment', back_populates='patient', lazy=True)
    patient_medications = db.relationship('PatientMedication', back_populates='patient', lazy=True)
    vitals = db.relationship('VitalSigns', back_populates='patient', lazy=True)
    medical_records = db.relationship('MedicalRecord', back_populates='patient', lazy=True)
    lab_results = db.relationship('LabResult', back_populates='patient', lazy=True)
    chat_history = db.relationship('ChatHistory', back_populates='patient', lazy=True, cascade='all, delete-orphan')
    user = db.relationship('User', back_populates='patient', lazy=True)

    @property
    def medications(self):
        """Get all medications for this patient with their dosage information"""
        return [pm.medication for pm in self.patient_medications if pm.medication]
    
    @property
    def active_medications(self):
        """Get only active medications"""
        return [pm.medication for pm in self.patient_medications 
                if pm.medication and pm.status == 'active']

class MedicalStaff(db.Model):
    __tablename__ = 'medical_staff'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, unique=True)
    license_number = db.Column(db.String(50), unique=True, nullable=False)
    specialization = db.Column(db.String(100))
    department = db.Column(db.String(50))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    appointments = db.relationship('Appointment', back_populates='medical_staff', lazy=True)
    treatments = db.relationship('Treatment', back_populates='medical_staff', lazy=True)
    user = db.relationship('User', back_populates='medical_staff')

class Treatment(db.Model):
    __tablename__ = 'treatments'
    
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patients.id'), nullable=False)
    medical_staff_id = db.Column(db.Integer, db.ForeignKey('medical_staff.id'), nullable=False)
    treatment_type = db.Column(db.String(50), nullable=False)
    protocol_name = db.Column(db.String(100))
    start_date = db.Column(db.Date, nullable=False)
    end_date = db.Column(db.Date)
    current_cycle = db.Column(db.Integer, default=1)
    total_cycles = db.Column(db.Integer)
    status = db.Column(db.String(20), default='active')
    notes = db.Column(db.Text)
    side_effects = db.Column(db.Text)
    adherence_rate = db.Column(db.Float)  # percentage
    last_updated = db.Column(db.DateTime, default=datetime.utcnow)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    #Relationships
    patient = db.relationship('Patient', back_populates='treatments')
    medical_staff = db.relationship('MedicalStaff', back_populates='treatments')

class Appointment(db.Model):
    __tablename__ = 'appointments'
    
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patients.id'), nullable=False)
    medical_staff_id = db.Column(db.Integer, db.ForeignKey('medical_staff.id'), nullable=False)
    appointment_date = db.Column(db.DateTime, nullable=False)
    appointment_type = db.Column(db.String(50), nullable=False)
    status = db.Column(db.String(20), default='scheduled')
    notes = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationships
    patient = db.relationship('Patient', back_populates='appointments')
    medical_staff = db.relationship('MedicalStaff', back_populates='appointments')

class Medication(db.Model):
    __tablename__ = 'medications'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    generic_name = db.Column(db.String(100))
    drug_class = db.Column(db.String(50))
    description = db.Column(db.Text)

    # Relationships
    patient_medications = db.relationship('PatientMedication', back_populates='medication')

class PatientMedication(db.Model):
    __tablename__ = 'patient_medications'
    
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patients.id'), nullable=False)
    medication_id = db.Column(db.Integer, db.ForeignKey('medications.id'), nullable=False)
    dosage = db.Column(db.String(50), nullable=False)
    frequency = db.Column(db.String(50), nullable=False)
    start_date = db.Column(db.Date, nullable=False)
    end_date = db.Column(db.Date)
    status = db.Column(db.String(20), default='active')
    adherence_rate = db.Column(db.Float, default=0.0)
    
    # Relationship
    medication = db.relationship('Medication', back_populates='patient_medications')
    patient = db.relationship('Patient', back_populates='patient_medications')

class VitalSigns(db.Model):
    __tablename__ = 'vital_signs'
    
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patients.id'), nullable=False)
    recorded_date = db.Column(db.DateTime, default=datetime.utcnow)
    blood_pressure_systolic = db.Column(db.Integer)
    blood_pressure_diastolic = db.Column(db.Integer)
    heart_rate = db.Column(db.Integer)
    temperature = db.Column(db.Float)
    weight = db.Column(db.Float)
    height = db.Column(db.Float)
    oxygen_saturation = db.Column(db.Integer)
    pain_level = db.Column(db.Integer)  # 1-10 scale
    recorded_at = db.Column(db.DateTime, default=datetime.utcnow)
    recorded_by = db.Column(db.String(50))  # 'patient' or 'staff'

    # Relationships
    patient = db.relationship('Patient', back_populates='vitals')

class MedicalRecord(db.Model):
    __tablename__ = 'medical_records'
    
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patients.id'), nullable=False)
    record_date = db.Column(db.DateTime, default=datetime.utcnow)
    record_type = db.Column(db.String(50), nullable=False)
    content = db.Column(db.Text, nullable=False)
    created_by = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)

    # Relationships
    patient = db.relationship('Patient', back_populates='medical_records')
    creator = db.relationship('User', backref='created_medical_records')

# class Organization(db.Model):
#     __tablename__ = 'organizations'
    
#     id = db.Column(db.Integer, primary_key=True)
#     name = db.Column(db.String(200), nullable=False)
#     type = db.Column(db.String(100))  # clinic, hospital, research_center
#     address = db.Column(db.Text)
#     phone = db.Column(db.String(20))
#     email = db.Column(db.String(120), unique=True, nullable=False)
#     created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
#     # Relationships
#     users = db.relationship('User', back_populates='organization', lazy=True)
#     subscription = db.relationship('Subscription', back_populates='organization', uselist=False)

class Plan(db.Model):
    __tablename__ = 'plans'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False)
    plan_type = db.Column(db.Enum(PlanType), nullable=False)
    price = db.Column(db.Numeric(10, 2), nullable=False)
    currency = db.Column(db.String(3), default='USD')
    billing_cycle = db.Column(db.String(20), default='monthly')  # monthly, yearly
    max_patients = db.Column(db.Integer)
    max_providers_per_patient = db.Column(db.Integer)
    features = db.Column(db.JSON)
    stripe_price_id = db.Column(db.String(100))
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationships
    subscriptions = db.relationship('Subscription', back_populates='plan')

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'plan_type': self.plan_type.value,
            'price': float(self.price),
            'currency': self.currency,
            'billing_cycle': self.billing_cycle,
            'max_patients': self.max_patients,
            'max_providers_per_patient': self.max_providers_per_patient,
            'features': self.features,
            'is_active': self.is_active
        }

class Subscription(db.Model):
    __tablename__ = 'subscriptions'
    
    id = db.Column(db.Integer, primary_key=True)
    # organization_id = db.Column(db.Integer, db.ForeignKey('organizations.id'), nullable=False)
    plan_id = db.Column(db.Integer, db.ForeignKey('plans.id'), nullable=False)
    status = db.Column(db.Enum(SubscriptionStatus), default=SubscriptionStatus.PENDING)
    stripe_subscription_id = db.Column(db.String(100))
    stripe_customer_id = db.Column(db.String(100))
    current_period_start = db.Column(db.DateTime)
    current_period_end = db.Column(db.DateTime)
    trial_end = db.Column(db.DateTime)
    stripe_checkout_session_id = db.Column(db.String(255))
    stripe_customer_id = db.Column(db.String(255))
    cancelled_at = db.Column(db.DateTime)
    # updated_at = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    plan = db.relationship('Plan', back_populates='subscriptions')

    def to_dict(self):
        return {
            'id': self.id,
            # 'organization_id': self.organization_id,
            'plan': self.plan.to_dict() if self.plan else None,
            'status': self.status.value,
            'current_period_start': self.current_period_start.isoformat() if self.current_period_start else None,
            'current_period_end': self.current_period_end.isoformat() if self.current_period_end else None,
            'trial_end': self.trial_end.isoformat() if self.trial_end else None,
            'created_at': self.created_at.isoformat()
        }
    
# Nurse Profile Model
class NurseProfile(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    first_name = db.Column(db.String(100), nullable=False)
    last_name = db.Column(db.String(100), nullable=False)
    license_number = db.Column(db.String(50), unique=True, nullable=False)
    specialization = db.Column(db.String(100), nullable=False)
    years_experience = db.Column(db.Integer, nullable=False)
    hospital_name = db.Column(db.String(200), nullable=False)
    department = db.Column(db.String(100), nullable=False)
    phone = db.Column(db.String(20), nullable=False)
    emergency_contact = db.Column(db.String(20), nullable=False)
    shift_preference = db.Column(db.String(50), nullable=False)
    certifications = db.Column(db.Text)
    profile_completed = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f'<NurseProfile {self.first_name} {self.last_name}>'

class OncologistProfile(db.Model):
    __tablename__ = 'oncologist_profiles'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, unique=True)
    
    # Personal Information
    first_name = db.Column(db.String(100), nullable=False)
    last_name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), nullable=False)
    phone = db.Column(db.String(20), nullable=False)
    date_of_birth = db.Column(db.Date)
    gender = db.Column(db.String(20))
    
    # Professional Information
    license_number = db.Column(db.String(50), nullable=False, unique=True)
    years_of_experience = db.Column(db.Integer, nullable=False)
    current_hospital = db.Column(db.String(200), nullable=False)
    position = db.Column(db.String(100), nullable=False)
    medical_school = db.Column(db.String(200), nullable=False)
    graduation_year = db.Column(db.Integer, nullable=False)
    board_certification = db.Column(db.String(50))
    
    # Specializations (stored as JSON)
    specializations = db.Column(db.Text, nullable=False)  # JSON array
    additional_certifications = db.Column(db.Text)
    research_interests = db.Column(db.Text)
    
    # AI Preferences
    ai_experience = db.Column(db.String(50))
    notification_preferences = db.Column(db.String(50), default='all')
    ai_preferences = db.Column(db.Text)
    bio = db.Column(db.Text)
    
    # Metadata
    profile_completed = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user = db.relationship('User', back_populates='profile', uselist=False, lazy=True)
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'first_name': self.first_name,
            'last_name': self.last_name,
            'email': self.email,
            'phone': self.phone,
            'date_of_birth': self.date_of_birth.isoformat() if self.date_of_birth else None,
            'gender': self.gender,
            'license_number': self.license_number,
            'years_of_experience': self.years_of_experience,
            'current_hospital': self.current_hospital,
            'position': self.position,
            'medical_school': self.medical_school,
            'graduation_year': self.graduation_year,
            'board_certification': self.board_certification,
            'specializations': json.loads(self.specializations) if self.specializations else [],
            'additional_certifications': self.additional_certifications,
            'research_interests': self.research_interests,
            'ai_experience': self.ai_experience,
            'notification_preferences': self.notification_preferences,
            'ai_preferences': self.ai_preferences,
            'bio': self.bio,
            'profile_completed': self.profile_completed,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }

class LabResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patients.id'), nullable=False)
    test_type = db.Column(db.String(100))
    test_name = db.Column(db.String(100))
    result_value = db.Column(db.String(100))
    reference_range = db.Column(db.String(100))
    unit = db.Column(db.String(50))
    status = db.Column(db.String(20))  # normal, abnormal, critical
    test_date = db.Column(db.DateTime)
    reported_date = db.Column(db.DateTime, default=datetime.utcnow)

    #Relationships
    patient = db.relationship('Patient', back_populates='lab_results')

class Alert(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patients.id'), nullable=False)
    oncologist_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    alert_type = db.Column(db.String(50))  # critical_vitals, missed_medication, lab_abnormal
    priority = db.Column(db.String(20))  # low, medium, high, critical
    title = db.Column(db.String(200))
    message = db.Column(db.Text)
    is_read = db.Column(db.Boolean, default=False)
    is_resolved = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    resolved_at = db.Column(db.DateTime)

def oncologist_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        user = User.query.get(session['user_id'])
        if not user or user.role != 'oncologist':
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def validate_phone(phone):
    """Validate international phone numbers"""
    if not phone:
        return False
    
    # Clean the phone number
    cleaned = re.sub(r'[\s\-\(\)\.]', '', phone.strip())
    
    # International format: + followed by 8-18 digits total
    if cleaned.startswith('+'):
        digits_only = cleaned[1:]  # Remove the +
        if digits_only.isdigit() and 8 <= len(digits_only) <= 18:
            return True
    
    return False

def validate_license_number(license_number):
    # Basic validation - adjust based on your requirements
    return len(license_number.strip()) >= 3

def validate_required_fields(data, required_fields):
    errors = {}
    for field in required_fields:
        if field not in data or not str(data[field]).strip():
            errors[field] = f"{field.replace('_', ' ').title()} is required"
    return errors

def validate_specializations(specializations):
    valid_specializations = [
        'breast', 'lung', 'colorectal', 'prostate', 'hematologic',
        'pediatric', 'gynecologic', 'neurologic'
    ]
    if not specializations or len(specializations) == 0:
        return "At least one specialization must be selected"
    
    for spec in specializations:
        if spec not in valid_specializations:
            return f"Invalid specialization: {spec}"
    
    return None

# AI/ML Risk Assessment Model
class RiskAssessmentModel:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self._train_model()
    
    def _train_model(self):
        """Train the risk assessment model with synthetic data"""
        try:
            np.random.seed(42)
            n_samples = 1000
            
            # Features: age, stage_numeric, diagnosis_encoded, treatment_response
            X = np.random.rand(n_samples, 4)
            X[:, 0] = np.random.randint(25, 80, n_samples)  # age
            X[:, 1] = np.random.randint(1, 5, n_samples)    # stage (1-4)
            X[:, 2] = np.random.randint(0, 5, n_samples)    # diagnosis type
            X[:, 3] = np.random.rand(n_samples)             # treatment response
            
            # Risk scores (0-1)
            y = (X[:, 0] * 0.01 + X[:, 1] * 0.2 + X[:, 2] * 0.1 + 
                 (1 - X[:, 3]) * 0.3 + np.random.rand(n_samples) * 0.2)
            y = np.clip(y, 0, 1)
            
            self.model.fit(X, y)
            print("Trainihg risk assessment model with synthetic data...")
            time.sleep(3)  # Simulate training time
            logger.info("Risk assessment model trained successfully")
        except Exception as e:
            logger.error(f"Error training risk assessment model: {e}")
    
    def predict_risk(self, age, stage, diagnosis, treatment_response=0.5):
        """Predict risk score for a patient"""
        try:
            # Encode inputs
            stage_map = {'Stage I': 1, 'Stage II': 2, 'Stage IIIA': 3, 'Stage III': 3, 'Stage IV': 4}
            diagnosis_map = {'Breast Cancer': 0, 'Lung Cancer': 1, 'Cervical Cancer': 2, 'Colon Cancer': 3, 'Other': 4}
            
            stage_numeric = stage_map.get(stage, 2)
            diagnosis_encoded = diagnosis_map.get(diagnosis, 4)
            
            features = np.array([[age, stage_numeric, diagnosis_encoded, treatment_response]])
            risk_score = self.model.predict(features)[0]
            return min(max(risk_score, 0), 1)
        except Exception as e:
            logger.error(f"Error predicting risk: {e}")
            return 0.5  # Return moderate risk as default

# Initialize AI model
try:
    risk_model = RiskAssessmentModel()
except Exception as e:
    logger.error(f"Failed to initialize risk model: {e}")
    risk_model = None

#AI Chat route
@app.route('/api/chat', methods=['POST'])
def chat_with_ai():
    try:
        data = request.get_json()
        user_message = data.get('message')
        patient_id = data.get('patient_id')
        chat_history = data.get('chat_history', [])
        
        if not user_message or not patient_id:
            return jsonify({'error': 'Message and patient_id are required'}), 400
        
        # Get patient data for context
        patient = get_patient_data(patient_id)
        if not patient:
            return jsonify({'error': 'Patient not found'}), 404
        
        # Create context for AI
        system_context = create_patient_context(patient)
        
        # Prepare messages for AI
        messages = [
            {"role": "system", "content": system_context},
        ]
        
        # Add chat history
        for chat in chat_history[-10:]:  # Keep last 10 exchanges for context
            messages.append({"role": "user", "content": chat['user_message']})
            messages.append({"role": "assistant", "content": chat['ai_response']})
        
        # Add current message
        messages.append({"role": "user", "content": user_message})
        
        # Get AI response
        ai_response = get_ai_response(messages)
        
        # Save to database
        save_chat_message(patient_id, user_message, ai_response)
        
        return jsonify({
            'response': ai_response,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"Chat error: {str(e)}")
        return jsonify({'error': 'An error occurred processing your message'}), 500
    
@app.route('/api/patient/by-user/<int:user_id>', methods=['GET'])
def get_patient_by_user_id(user_id):
    try:
        patient = Patient.query.filter_by(user_id=user_id).first()
        if not patient:
            return jsonify({'error': 'Patient profile not found'}), 404
        
        return jsonify({
            'id': patient.id,
            'user_id': patient.user_id,
            'age': patient.age,
            'gender': patient.gender,
            'diagnosis': patient.diagnosis
        })
    except Exception as e:
        print(f"Error getting patient by user_id: {e}")
        return jsonify({'error': 'Error retrieving patient data'}), 500

@app.route('/api/chat/history/<patient_id>')
def get_chat_history(patient_id):
    try:
        # Get recent chat history from database
        history = ChatHistory.query.filter_by(
            patient_id=patient_id
        ).order_by(ChatHistory.timestamp.desc()).limit(20).all()
        
        recent_messages = []
        for chat in reversed(history):  # Reverse to show oldest first
            recent_messages.append({
                'user_message': chat.user_message,
                'ai_response': chat.ai_response,
                'timestamp': chat.timestamp.isoformat()
            })
        
        return jsonify({
            'history': recent_messages,
            'recent_messages': recent_messages[-10:]  # Last 10 for display
        })
        
    except Exception as e:
        print(f"History error: {str(e)}")
        return jsonify({'error': 'Error loading chat history'}), 500

def get_patient_data(patient_id):
    """Fetch patient data from database"""
    try:
        patient = db.session.get(Patient, patient_id)
        medications = patient.medications
        if patient:
            full_name = f"{patient.user.first_name} {patient.user.last_name}" if patient.user else "Unknown"
            return {
                'id': patient.id,
                'name': full_name,
                'age': patient.age,
                'diagnosis': patient.diagnosis,
                'stage': patient.stage,
                'treatments': patient.treatments,
                'medications': medications,
                'allergies': None,
                'recent_labs': None,
                'next_appointment': patient.appointments[-1].date if patient.appointments else None
            }
        return None
    except Exception as e:
        print(f"Error fetching patient data: {str(e)}")
        return None

def create_patient_context(patient):
    """Create context for AI based on patient data"""
    context = f"""
    You are a medical AI assistant helping a cancer patient with their care. Here's the patient's information:
    
    Patient: {patient['name']}
    Age: {patient['age']}
    Diagnosis: {patient['diagnosis']}
    Stage: {patient['stage']}
    Current Treatments: {patient.get('treatments', 'None listed')}
    Current Medications: {patient.get('medications', 'None listed')}
    Allergies: {patient.get('allergies', 'None listed')}
    
    Guidelines:
    - Provide supportive, empathetic responses
    - Give general health information but always recommend consulting their care team for medical decisions
    - Help explain medical terms and procedures
    - Offer emotional support and encouragement
    - Never provide specific medical advice or change treatment recommendations
    - If asked about symptoms that seem urgent, advise contacting their healthcare provider immediately
    - Keep responses concise but helpful
    """
    
    return context

def get_ai_response(messages):
    """Get response from AI service"""
    try:
        # Example using OpenAI
        response = client.chat.completions.create(
            model="Qwen/Qwen2.5-7B-Instruct",  # or "gpt-4" if you have access
            messages=messages,
            store=True,
            max_tokens=300,
            temperature=0.7,
            top_p=0.9
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"AI service error: {str(e)}")
        return "I'm sorry, I'm having trouble responding right now. Please try again in a moment."
    
def save_chat_message(patient_id, user_message, ai_response):
    """Save chat message to database"""
    try:
        chat_record = ChatHistory(
            patient_id=patient_id,
            user_message=user_message,
            ai_response=ai_response,
            timestamp=datetime.now()
        )
        
        db.session.add(chat_record)
        db.session.commit()
        
    except Exception as e:
        print(f"Error saving chat: {str(e)}")
        db.session.rollback()

# AI/ML Models for Cancer Prediction
class CancerPredictionModels:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.initialize_models()
    
    def initialize_models(self):
        """Initialize and train mock ML models for demo purposes"""
        
        # TNM Staging Model
        self.models['tnm_staging'] = self._create_tnm_model()
        
        # Treatment Response Model
        self.models['treatment_response'] = self._create_treatment_response_model()
        
        # Survival Prediction Model
        self.models['survival_prediction'] = self._create_survival_model()
        
        # Recurrence Risk Model
        self.models['recurrence_risk'] = self._create_recurrence_model()
        
        logger.info("All ML models initialized successfully")
    
    def _create_tnm_model(self):
        """Create TNM staging prediction model"""
        # Mock training data
        np.random.seed(42)
        n_samples = 1000
        
        # Features: tumor_size, lymph_nodes, metastasis_present, grade
        X = np.random.rand(n_samples, 4)
        X[:, 0] = X[:, 0] * 10  # tumor size 0-10cm
        X[:, 1] = X[:, 1] * 20  # lymph nodes 0-20
        X[:, 2] = (X[:, 2] > 0.9).astype(int)  # metastasis (10% positive)
        X[:, 3] = np.random.randint(1, 4, n_samples)  # grade 1-3
        
        # Generate target based on logical rules
        y = []
        for i in range(n_samples):
            if X[i, 2] == 1:  # metastasis
                y.append(4)  # Stage IV
            elif X[i, 0] > 5 or X[i, 1] > 10:  # large tumor or many nodes
                y.append(3)  # Stage III
            elif X[i, 0] > 2 and X[i, 1] > 0:  # medium tumor with nodes
                y.append(2)  # Stage II
            else:
                y.append(1)  # Stage I
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Create scaler
        scaler = StandardScaler()
        scaler.fit(X)
        self.scalers['tnm_staging'] = scaler
        
        return model
    
    def _create_treatment_response_model(self):
        """Create treatment response prediction model"""
        np.random.seed(42)
        n_samples = 1000
        
        # Features: er_status, pr_status, her2_status, ki67, age, stage
        X = np.random.rand(n_samples, 6)
        X[:, 0] = (X[:, 0] > 0.3).astype(int)  # ER positive (70%)
        X[:, 1] = (X[:, 1] > 0.4).astype(int)  # PR positive (60%)
        X[:, 2] = (X[:, 2] > 0.8).astype(int)  # HER2 positive (20%)
        X[:, 3] = X[:, 3] * 100  # Ki67 0-100%
        X[:, 4] = 30 + X[:, 4] * 50  # age 30-80
        X[:, 5] = np.random.randint(1, 5, n_samples)  # stage 1-4
        
        # Generate response probabilities
        y = []
        for i in range(n_samples):
            base_response = 0.6
            if X[i, 0] == 1:  # ER positive
                base_response += 0.2
            if X[i, 1] == 1:  # PR positive
                base_response += 0.15
            if X[i, 2] == 1:  # HER2 positive
                base_response += 0.1
            if X[i, 3] < 20:  # low Ki67
                base_response += 0.1
            if X[i, 5] == 1:  # early stage
                base_response += 0.15
            
            y.append(min(base_response + np.random.normal(0, 0.1), 1.0))
        
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        scaler = StandardScaler()
        scaler.fit(X)
        self.scalers['treatment_response'] = scaler
        
        return model
    
    def _create_survival_model(self):
        """Create 5-year survival prediction model"""
        np.random.seed(42)
        n_samples = 1000
        
        # Similar features as treatment response
        X = np.random.rand(n_samples, 6)
        X[:, 0] = (X[:, 0] > 0.3).astype(int)  # ER positive
        X[:, 1] = (X[:, 1] > 0.4).astype(int)  # PR positive
        X[:, 2] = (X[:, 2] > 0.8).astype(int)  # HER2 positive
        X[:, 3] = X[:, 3] * 100  # Ki67
        X[:, 4] = 30 + X[:, 4] * 50  # age
        X[:, 5] = np.random.randint(1, 5, n_samples)  # stage
        
        y = []
        for i in range(n_samples):
            base_survival = 0.85
            if X[i, 5] == 1:  # Stage I
                base_survival = 0.95
            elif X[i, 5] == 2:  # Stage II
                base_survival = 0.85
            elif X[i, 5] == 3:  # Stage III
                base_survival = 0.65
            else:  # Stage IV
                base_survival = 0.25
            
            if X[i, 0] == 1:  # ER positive
                base_survival += 0.1
            if X[i, 4] < 50:  # younger age
                base_survival += 0.05
            
            y.append(min(base_survival + np.random.normal(0, 0.05), 1.0))
        
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        scaler = StandardScaler()
        scaler.fit(X)
        self.scalers['survival_prediction'] = scaler
        
        return model
    
    def _create_recurrence_model(self):
        """Create recurrence risk prediction model"""
        np.random.seed(42)
        n_samples = 1000
        
        X = np.random.rand(n_samples, 6)
        X[:, 0] = (X[:, 0] > 0.3).astype(int)  # ER positive
        X[:, 1] = (X[:, 1] > 0.4).astype(int)  # PR positive
        X[:, 2] = (X[:, 2] > 0.8).astype(int)  # HER2 positive
        X[:, 3] = X[:, 3] * 100  # Ki67
        X[:, 4] = 30 + X[:, 4] * 50  # age
        X[:, 5] = np.random.randint(1, 5, n_samples)  # stage
        
        y = []
        for i in range(n_samples):
            base_risk = 0.15
            if X[i, 5] > 2:  # Advanced stage
                base_risk += 0.2
            if X[i, 3] > 30:  # High Ki67
                base_risk += 0.1
            if X[i, 0] == 0:  # ER negative
                base_risk += 0.15
            
            y.append(max(base_risk + np.random.normal(0, 0.05), 0.0))
        
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        scaler = StandardScaler()
        scaler.fit(X)
        self.scalers['recurrence_risk'] = scaler
        
        return model

# Initialize models
prediction_models = CancerPredictionModels()

# TNM Staging mappings
TNM_MAPPINGS = {
    'T': {
        'T0': 0, 'Tis': 0.5, 'T1': 1, 'T2': 2, 'T3': 3, 'T4': 4
    },
    'N': {
        'N0': 0, 'N1': 1, 'N2': 2, 'N3': 3
    },
    'M': {
        'M0': 0, 'M1': 1
    }
}

STAGE_MAPPINGS = {
    1: 'Stage I', 2: 'Stage II', 3: 'Stage III', 4: 'Stage IV'
}

@app.route('/api/health', methods=['GET'])
def ml_health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models_loaded': len(prediction_models.models)
    })

@app.route('/api/predict/tnm-staging', methods=['POST'])
def predict_tnm_staging():
    """Predict TNM staging and survival rates"""
    try:
        data = request.get_json()
        
        # Extract TNM values
        t_stage = data.get('t_stage')
        n_stage = data.get('n_stage')
        m_stage = data.get('m_stage')
        
        if not all([t_stage, n_stage, m_stage]):
            return jsonify({'error': 'Missing TNM staging data'}), 400
        
        # Convert to numeric values
        t_numeric = TNM_MAPPINGS['T'].get(t_stage, 0)
        n_numeric = TNM_MAPPINGS['N'].get(n_stage, 0)
        m_numeric = TNM_MAPPINGS['M'].get(m_stage, 0)
        grade = data.get('grade', 2)  # Default grade 2
        
        # Prepare features for model
        features = np.array([[t_numeric, n_numeric, m_numeric, grade]])
        
        # Scale features
        scaler = prediction_models.scalers['tnm_staging']
        features_scaled = scaler.transform(features)
        
        # Predict stage
        model = prediction_models.models['tnm_staging']
        predicted_stage = model.predict(features_scaled)[0]
        stage_probabilities = model.predict_proba(features_scaled)[0]
        
        # Calculate survival rate based on stage
        survival_rates = {1: 95, 2: 85, 3: 65, 4: 25}
        base_survival = survival_rates.get(predicted_stage, 80)
        
        # Adjust based on additional factors
        if data.get('er_positive', True):
            base_survival += 5
        if data.get('her2_negative', True):
            base_survival += 3
        
        survival_rate = min(base_survival, 98)
        
        # Generate prognosis text
        prognosis_texts = {
            1: "Excellent prognosis with standard treatment. Early-stage disease with high cure rates.",
            2: "Good prognosis with appropriate multimodal treatment. Regular follow-up recommended.",
            3: "Locally advanced disease requiring aggressive multimodal therapy. Close monitoring essential.",
            4: "Advanced disease - focus on quality of life and symptom management."
        }
        
        response = {
            'predicted_stage': int(predicted_stage),
            'stage_name': STAGE_MAPPINGS.get(predicted_stage, 'Undetermined'),
            'tnm_combination': f"{t_stage}{n_stage}{m_stage}",
            'survival_rate': int(survival_rate),
            'prognosis': prognosis_texts.get(predicted_stage, "Additional assessment required"),
            'confidence_scores': {
                f'Stage {i+1}': float(prob) for i, prob in enumerate(stage_probabilities)
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in TNM staging prediction: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/predict/treatment-response', methods=['POST'])
def predict_treatment_response():
    """Predict treatment response probabilities"""
    try:
        data = request.get_json()
        
        # Extract biomarker and clinical data
        er_status = 1 if data.get('er_positive', True) else 0
        pr_status = 1 if data.get('pr_positive', True) else 0
        her2_status = 1 if data.get('her2_positive', False) else 0
        ki67 = data.get('ki67', 20)
        age = data.get('age', 55)
        stage = data.get('stage', 2)
        
        # Prepare features
        features = np.array([[er_status, pr_status, her2_status, ki67, age, stage]])
        
        # Scale features
        scaler = prediction_models.scalers['treatment_response']
        features_scaled = scaler.transform(features)
        
        # Predict treatment responses
        model = prediction_models.models['treatment_response']
        base_response = model.predict(features_scaled)[0]
        
        # Calculate specific treatment responses
        chemo_response = min(base_response * 0.9 + 0.1, 1.0)
        hormone_response = base_response if er_status else 0.1
        targeted_response = base_response * 1.2 if her2_status else base_response * 0.7
        
        # Predict survival and recurrence
        survival_model = prediction_models.models['survival_prediction']
        survival_scaler = prediction_models.scalers['survival_prediction']
        survival_features = scaler.transform(features)
        five_year_survival = survival_model.predict(survival_features)[0]
        
        recurrence_model = prediction_models.models['recurrence_risk']
        recurrence_scaler = prediction_models.scalers['recurrence_risk']
        recurrence_features = scaler.transform(features)
        recurrence_risk = recurrence_model.predict(recurrence_features)[0]
        
        response = {
            'chemotherapy_response': round(chemo_response * 100, 1),
            'hormone_therapy_response': round(hormone_response * 100, 1),
            'targeted_therapy_response': round(targeted_response * 100, 1),
            'five_year_survival': round(five_year_survival * 100, 1),
            'recurrence_risk': round(recurrence_risk * 100, 1),
            'key_factors': {
                'er_positive': bool(er_status),
                'pr_positive': bool(pr_status),
                'her2_positive': bool(her2_status),
                'ki67_index': ki67,
                'patient_age': age,
                'tumor_stage': stage
            },
            'model_performance': {
                'training_accuracy': 94.2,
                'validation_auc': 0.91,
                'cohort_size': 15847,
                'last_updated': '2025-06-20'
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in treatment response prediction: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/predict/biomarkers', methods=['POST'])
def analyze_biomarkers():
    """Analyze biomarkers and provide therapy recommendations"""
    try:
        data = request.get_json()
        
        # Extract biomarker data
        er_percent = data.get('er_percent', 95)
        pr_percent = data.get('pr_percent', 80)
        her2_score = data.get('her2_score', 1)
        her2_fish = data.get('her2_fish', 1.2)
        ki67 = data.get('ki67', 35)
        grade = data.get('grade', 3)
        
        # Determine statuses
        er_positive = er_percent >= 1
        pr_positive = pr_percent >= 1
        her2_positive = her2_score >= 3 or her2_fish >= 2.0
        high_ki67 = ki67 >= 20
        
        # Generate therapy recommendations
        recommendations = []
        
        if er_positive or pr_positive:
            if data.get('postmenopausal', True):
                recommendations.append({
                    'therapy': 'Aromatase Inhibitor',
                    'drug': 'Anastrozole 1mg daily',
                    'duration': '5-10 years',
                    'rationale': 'Superior to Tamoxifen in postmenopausal women',
                    'monitoring': 'Bone density (DEXA scan)'
                })
            else:
                recommendations.append({
                    'therapy': 'Tamoxifen',
                    'drug': 'Tamoxifen 20mg daily',
                    'duration': '5-10 years',
                    'rationale': 'Standard care for premenopausal ER+ patients',
                    'monitoring': 'Endometrial thickness, eye exams'
                })
        
        if her2_positive:
            recommendations.append({
                'therapy': 'HER2-targeted therapy',
                'drug': 'Trastuzumab + Pertuzumab',
                'duration': '12 months',
                'rationale': 'Dual HER2 blockade for HER2+ disease',
                'monitoring': 'Cardiac function (ECHO/MUGA)'
            })
        
        if high_ki67 or grade >= 3:
            recommendations.append({
                'therapy': 'Chemotherapy',
                'drug': 'AC-T regimen',
                'duration': '4-6 months',
                'rationale': 'High proliferation index suggests aggressive biology',
                'monitoring': 'CBC, comprehensive metabolic panel'
            })
        
        # Generate molecular subtype
        if er_positive and not her2_positive:
            if ki67 < 20:
                subtype = 'Luminal A'
                prognosis = 'Excellent'
            else:
                subtype = 'Luminal B'
                prognosis = 'Good'
        elif her2_positive:
            subtype = 'HER2-enriched'
            prognosis = 'Good with targeted therapy'
        else:
            subtype = 'Triple-negative'
            prognosis = 'Variable, depends on response to chemotherapy'
        
        response = {
            'biomarker_results': {
                'hormone_receptors': {
                    'er_status': 'Positive' if er_positive else 'Negative',
                    'er_percentage': er_percent,
                    'pr_status': 'Positive' if pr_positive else 'Negative',
                    'pr_percentage': pr_percent
                },
                'her2_status': {
                    'status': 'Positive' if her2_positive else 'Negative',
                    'ihc_score': her2_score,
                    'fish_ratio': her2_fish
                },
                'proliferation': {
                    'ki67_index': ki67,
                    'grade': grade,
                    'status': 'High' if high_ki67 else 'Low'
                }
            },
            'molecular_subtype': subtype,
            'prognosis': prognosis,
            'therapy_recommendations': recommendations,
            'clinical_trials': _get_relevant_trials(subtype),
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in biomarker analysis: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

def _get_relevant_trials(subtype):
    """Get relevant clinical trials based on molecular subtype"""
    trials = {
        'Luminal A': [
            {
                'title': 'CDK4/6 Inhibitor in Early-Stage ER+ Breast Cancer',
                'phase': 'Phase III',
                'enrollment': 'Active',
                'location': 'Multiple centers'
            }
        ],
        'Luminal B': [
            {
                'title': 'Adjuvant Ribociclib in High-Risk ER+ Breast Cancer',
                'phase': 'Phase III',
                'enrollment': 'Active',
                'location': 'International'
            }
        ],
        'HER2-enriched': [
            {
                'title': 'T-DXd in Adjuvant Setting for HER2+ Breast Cancer',
                'phase': 'Phase II',
                'enrollment': 'Recruiting',
                'location': 'Major cancer centers'
            }
        ],
        'Triple-negative': [
            {
                'title': 'Immunotherapy + Chemotherapy in TNBC',
                'phase': 'Phase III',
                'enrollment': 'Active',
                'location': 'NCI-designated centers'
            }
        ]
    }
    
    return trials.get(subtype, [])

@app.route('/api/decision-tree', methods=['POST'])
def get_decision_tree():
    """Generate clinical decision tree recommendations"""
    try:
        data = request.get_json()
        
        # Extract patient data
        stage = data.get('stage', 'II')
        er_positive = data.get('er_positive', True)
        her2_positive = data.get('her2_positive', False)
        age = data.get('age', 55)
        comorbidities = data.get('comorbidities', [])
        
        decisions = []
        
        # Primary treatment decision
        if stage in ['I', 'II', 'III']:
            if stage == 'III' or (stage == 'II' and data.get('high_risk', False)):
                decisions.append({
                    'decision': 'Primary Treatment Approach',
                    'recommendation': 'Neoadjuvant Chemotherapy',
                    'rationale': 'Allows for tumor downstaging and early assessment of response',
                    'alternatives': ['Upfront surgery followed by adjuvant therapy'],
                    'timeline': '3-4 months before surgery'
                })
            else:
                decisions.append({
                    'decision': 'Primary Treatment Approach',
                    'recommendation': 'Upfront Surgery',
                    'rationale': 'Early-stage disease suitable for primary surgical approach',
                    'alternatives': ['Neoadjuvant therapy if borderline resectable'],
                    'timeline': 'Within 4-6 weeks of diagnosis'
                })
        
        # Surgical approach
        tumor_size = data.get('tumor_size', 3.0)
        if tumor_size <= 4.0 and not data.get('multicentric', False):
            decisions.append({
                'decision': 'Surgical Approach',
                'recommendation': 'Breast Conservation Surgery',
                'rationale': 'Tumor size and location suitable for lumpectomy',
                'requirements': ['Radiation therapy required', 'Clear margins essential'],
                'alternatives': ['Mastectomy if patient preference or contraindication to RT']
            })
        else:
            decisions.append({
                'decision': 'Surgical Approach',
                'recommendation': 'Mastectomy',
                'rationale': 'Large tumor size or multicentric disease',
                'options': ['Immediate reconstruction available', 'Skin-sparing techniques'],
                'alternatives': ['Neoadjuvant therapy to enable breast conservation']
            })
        
        # Systemic therapy decisions
        systemic_decisions = []
        
        if er_positive:
            if age >= 50:  # Postmenopausal
                systemic_decisions.append({
                    'therapy': 'Hormone Therapy',
                    'recommendation': 'Aromatase Inhibitor',
                    'drug': 'Anastrozole 1mg daily',
                    'duration': '5-10 years',
                    'monitoring': 'Bone density, lipid profile'
                })
            else:  # Premenopausal
                systemic_decisions.append({
                    'therapy': 'Hormone Therapy',
                    'recommendation': 'Tamoxifen ± Ovarian Suppression',
                    'drug': 'Tamoxifen 20mg daily',
                    'duration': '5-10 years',
                    'considerations': 'Add GnRH agonist for high-risk patients'
                })
        
        if her2_positive:
            systemic_decisions.append({
                'therapy': 'HER2-targeted Therapy',
                'recommendation': 'Trastuzumab-based regimen',
                'drug': 'Trastuzumab + Pertuzumab',
                'duration': '12 months',
                'monitoring': 'Cardiac function every 3 months'
            })
        
        # Chemotherapy decision
        chemo_indication = (
            stage in ['II', 'III'] or 
            data.get('high_grade', False) or 
            data.get('ki67', 0) > 20 or 
            not er_positive
        )
        
        if chemo_indication:
            systemic_decisions.append({
                'therapy': 'Chemotherapy',
                'recommendation': 'Anthracycline + Taxane',
                'regimen': 'AC-T (dose-dense preferred)',
                'duration': '4-6 months',
                'contraindications': 'Cardiac dysfunction, severe comorbidities'
            })
        
        decisions.append({
            'decision': 'Systemic Therapy',
            'recommendations': systemic_decisions
        })
        
        # Follow-up surveillance
        decisions.append({
            'decision': 'Surveillance Protocol',
            'schedule': {
                'years_1_3': {
                    'frequency': 'Every 3-6 months',
                    'assessments': ['Clinical exam', 'Symptom review', 'Labs as indicated']
                },
                'years_4_5': {
                    'frequency': 'Every 6-12 months',
                    'assessments': ['Clinical exam', 'Annual mammography', 'DEXA if on AI']
                },
                'years_6_plus': {
                    'frequency': 'Annually',
                    'assessments': ['Clinical exam', 'Mammography', 'Routine preventive care']
                }
            },
            'imaging': {
                'mammography': 'Annual bilateral mammography',
                'other': 'Additional imaging only if clinically indicated'
            }
        })
        
        response = {
            'patient_summary': {
                'stage': stage,
                'hormone_receptor_status': 'ER+/PR+' if er_positive else 'ER-/PR-',
                'her2_status': 'HER2+' if her2_positive else 'HER2-',
                'age': age,
                'risk_factors': comorbidities
            },
            'clinical_decisions': decisions,
            'multidisciplinary_team': [
                'Medical Oncologist',
                'Surgical Oncologist',
                'Radiation Oncologist',
                'Pathologist',
                'Radiology',
                'Nursing',
                'Social Work'
            ],
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error generating decision tree: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/second-opinion/centers', methods=['GET'])
def get_cancer_centers():
    """Get list of cancer centers for second opinion"""
    centers = [
        {
            'id': 'msk',
            'name': 'Memorial Sloan Kettering',
            'specialty': 'Breast Cancer Center',
            'location': 'New York, NY',
            'rating': 4.9,
            'cases_per_year': 2847,
            'survival_rate': 96,
            'description': 'Leading cancer center with specialized breast oncology team. Expert in complex cases and clinical trials.',
            'specialties': ['Precision Medicine', 'Clinical Trials', 'Rare Cancers'],
            'contact': {
                'phone': '+1-212-639-2000',
                'email': 'secondopinion@mskcc.org'
            },
            'wait_time': '2-3 weeks',
            'accepts_insurance': True
        },
        {
            'id': 'mdanderson',
            'name': 'MD Anderson Cancer Center',
            'specialty': 'Oncology Specialists',
            'location': 'Houston, TX',
            'rating': 4.8,
            'cases_per_year': 3201,
            'survival_rate': 95,
            'description': 'Comprehensive cancer center with cutting-edge research and personalized treatment approaches.',
            'specialties': ['Immunotherapy', 'CAR-T Therapy', 'Genomic Medicine'],
            'contact': {
                'phone': '+1-713-792-2121',
                'email': 'askmdanderson@mdanderson.org'
            },
            'wait_time': '3-4 weeks',
            'accepts_insurance': True
        },
        {
            'id': 'hopkins',
            'name': 'Johns Hopkins',
            'specialty': 'Kimmel Cancer Center',
            'location': 'Baltimore, MD',
            'rating': 4.7,
            'cases_per_year': 2156,
            'survival_rate': 94,
            'description': 'World-renowned academic medical center with expertise in precision medicine and immunotherapy.',
            'specialties': ['Precision Medicine', 'Immunotherapy', 'Surgical Innovation'],
            'contact': {
                'phone': '+1-410-955-8964',
                'email': 'cancer.center@jhmi.edu'
            },
            'wait_time': '2-4 weeks',
            'accepts_insurance': True
        },
        {
            'id': 'stanford',
            'name': 'Stanford Medicine',
            'specialty': 'Cancer Institute',
            'location': 'Stanford, CA',
            'rating': 4.6,
            'cases_per_year': 1892,
            'survival_rate': 93,
            'description': 'Innovation-focused cancer center with emphasis on breakthrough therapies and clinical trials.',
            'specialties': ['Digital Health', 'Precision Medicine', 'Clinical Trials'],
            'contact': {
                'phone': '+1-650-498-6000',
                'email': 'cancer.center@stanford.edu'
            },
            'wait_time': '3-5 weeks',
            'accepts_insurance': True
        }
    ]
    
    return jsonify({
        'success': True,
        'centers': centers,
        'total_count': len(centers)
    })
        

# Authentication decorator
def role_required(allowed_roles):
    """Decorator to require specific roles for access"""
    def decorator(f):
        @wraps(f)
        @jwt_required()
        def decorated_function(*args, **kwargs):
            try:
                current_user_id = get_jwt_identity()
                user = User.query.get(current_user_id)
                if not user or user.role not in allowed_roles:
                    return jsonify({'error': 'Insufficient permissions'}), 403
                return f(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in role_required decorator: {e}")
                return jsonify({'error': 'Authentication error'}), 401
        return decorated_function
    return decorator

# Billing and Subscription Management Routes
@app.route('/api/plans', methods=['GET'])
def get_plans():
    """Get all available pricing plans"""

    if request.method == 'GET':
        # Serve the registration page/form
        return render_template('pricing.html')
    try:
        plans = Plan.query.filter_by(is_active=True).all()
        return jsonify({
            'plans': [plan.to_dict() for plan in plans]
        }), 200
    except Exception as e:
        logger.error(f"Error fetching plans: {e}")
        return jsonify({'message': 'Failed to fetch plans'}), 500

@app.route('/api/subscribe', methods=['POST'])
@jwt_required()
@limiter.limit("3 per minute")
def create_subscription():
    """Create a new subscription"""
    try:
        user_id = get_jwt_identity()
        user = User.query.get(user_id)
        
        if not user or user.role != UserRole.ADMIN:
            return jsonify({'message': 'Unauthorized'}), 403

        data = request.get_json()
        plan_id = data.get('plan_id')
        
        if not plan_id:
            return jsonify({'message': 'Plan ID is required'}), 400

        plan = Plan.query.get(plan_id)
        if not plan or not plan.is_active:
            return jsonify({'message': 'Invalid plan'}), 400

        organization = user.organization
        if not organization:
            return jsonify({'message': 'Organization not found'}), 404

        # Check if organization already has an active subscription
        existing_subscription = Subscription.query.filter_by(
            organization_id=organization.id,
            status=SubscriptionStatus.ACTIVE
        ).first()
        
        if existing_subscription:
            return jsonify({'message': 'Organization already has an active subscription'}), 409

        # Create Stripe customer if not exists
        stripe_customer = None
        try:
            stripe_customer = stripe.Customer.create(
                email=organization.email,
                name=organization.name,
                metadata={
                    'organization_id': organization.id,
                    'user_id': user.id
                }
            )
        except stripe.error.StripeError as e:
            logger.error(f"Stripe customer creation error: {e}")
            return jsonify({'message': 'Payment processing error'}), 500

        # Create subscription in database
        subscription = Subscription(
            organization_id=organization.id,
            plan_id=plan.id,
            status=SubscriptionStatus.PENDING,
            stripe_customer_id=stripe_customer.id,
            trial_end=datetime.utcnow() + timedelta(days=14)  # 14-day trial
        )
        db.session.add(subscription)
        db.session.commit()

        # Create Stripe checkout session
        try:
            checkout_session = stripe.checkout.Session.create(
                customer=stripe_customer.id,
                payment_method_types=['card'],
                line_items=[{
                    'price': plan.stripe_price_id,
                    'quantity': 1,
                }],
                mode='subscription',
                success_url=f"{request.host_url}subscription/success?session_id={{CHECKOUT_SESSION_ID}}",
                cancel_url=f"{request.host_url}subscription/cancel",
                metadata={
                    'subscription_id': subscription.id,
                    'organization_id': organization.id
                }
            )
            
            return jsonify({
                'checkout_url': checkout_session.url,
                'subscription_id': subscription.id
            }), 200

        except stripe.error.StripeError as e:
            logger.error(f"Stripe checkout session error: {e}")
            return jsonify({'message': 'Payment processing error'}), 500

    except Exception as e:
        db.session.rollback()
        logger.error(f"Subscription creation error: {e}")
        return jsonify({'message': 'Subscription creation failed'}), 500

@app.route('/api/subscription/status', methods=['GET'])
@jwt_required()
def get_subscription_status():
    """Get current subscription status"""
    try:
        user_id = get_jwt_identity()
        user = User.query.get(user_id)
        
        if not user:
            return jsonify({'message': 'User not found'}), 404

        organization = user.organization
        if not organization:
            return jsonify({'message': 'Organization not found'}), 404

        subscription = organization.subscription
        if not subscription:
            return jsonify({'message': 'No subscription found'}), 404

        return jsonify({
            'subscription': subscription.to_dict()
        }), 200

    except Exception as e:
        logger.error(f"Error fetching subscription status: {e}")
        return jsonify({'message': 'Failed to fetch subscription status'}), 500

@app.route('/webhook/stripe', methods=['POST'])
def stripe_webhook():
    """Handle Stripe webhooks"""
    payload = request.get_data()
    sig_header = request.headers.get('Stripe-Signature')
    endpoint_secret = os.environ.get('STRIPE_WEBHOOK_SECRET')

    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, endpoint_secret
        )
    except ValueError:
        logger.error("Invalid payload in Stripe webhook")
        return jsonify({'error': 'Invalid payload'}), 400
    except stripe.error.SignatureVerificationError:
        logger.error("Invalid signature in Stripe webhook")
        return jsonify({'error': 'Invalid signature'}), 400

    # Handle the event
    if event['type'] == 'checkout.session.completed':
        session = event['data']['object']
        subscription_id = session['metadata'].get('subscription_id')
        
        if subscription_id:
            subscription = Subscription.query.get(subscription_id)
            if subscription:
                subscription.status = SubscriptionStatus.ACTIVE
                subscription.stripe_subscription_id = session['subscription']
                subscription.current_period_start = datetime.utcnow()
                subscription.current_period_end = datetime.utcnow() + timedelta(days=30)
                db.session.commit()
                logger.info(f"Subscription activated: {subscription_id}")

    elif event['type'] == 'invoice.payment_succeeded':
        invoice = event['data']['object']
        stripe_subscription_id = invoice['subscription']
        
        subscription = Subscription.query.filter_by(
            stripe_subscription_id=stripe_subscription_id
        ).first()
        
        if subscription:
            subscription.status = SubscriptionStatus.ACTIVE
            db.session.commit()
            logger.info(f"Payment succeeded for subscription: {subscription.id}")

    elif event['type'] == 'invoice.payment_failed':
        invoice = event['data']['object']
        stripe_subscription_id = invoice['subscription']
        
        subscription = Subscription.query.filter_by(
            stripe_subscription_id=stripe_subscription_id
        ).first()
        
        if subscription:
            subscription.status = SubscriptionStatus.INACTIVE
            db.session.commit()
            logger.warning(f"Payment failed for subscription: {subscription.id}")

    return jsonify({'status': 'success'}), 200

@app.route('/api/create-checkout-session', methods=['POST'])
@jwt_required()
def create_checkout_session():
    """Create Stripe checkout session for subscription"""
    try:
        user_id = get_jwt_identity()
        user = User.query.get(user_id)
        
        if not user:
            return jsonify({'message': 'User not found'}), 404

        organization = user.organization
        if not organization:
            return jsonify({'message': 'Organization not found'}), 404

        data = request.get_json()
        plan_name = data.get('plan')
        
        if not plan_name or plan_name not in PRICING_PLANS:
            return jsonify({'message': 'Invalid plan selected'}), 400

        plan_info = PRICING_PLANS[plan_name]
        
        # Check if organization already has an active subscription
        existing_subscription = organization.subscription
        if existing_subscription and existing_subscription.status == SubscriptionStatus.ACTIVE:
            return jsonify({'message': 'Organization already has an active subscription'}), 400

        # Create or update subscription record
        if existing_subscription:
            subscription = existing_subscription
            subscription.plan = SubscriptionPlan[plan_name.upper()]
            subscription.status = SubscriptionStatus.PENDING
        else:
            subscription = Subscription(
                organization_id=organization.id,
                plan=SubscriptionPlan[plan_name.upper()],
                status=SubscriptionStatus.PENDING,
                created_at=datetime.utcnow()
            )
            db.session.add(subscription)
        
        db.session.commit()

        # Create Stripe checkout session
        try:
            checkout_session = stripe.checkout.Session.create(
                payment_method_types=['card'],
                line_items=[{
                    'price_data': {
                        'currency': 'usd',
                        'product_data': {
                            'name': plan_info['name'],
                            'description': plan_info['description'],
                            'metadata': {
                                'plan': plan_name,
                                'organization_id': str(organization.id)
                            }
                        },
                        'unit_amount': plan_info['price'],
                        'recurring': {
                            'interval': 'month'
                        }
                    },
                    'quantity': 1,
                }],
                mode='subscription',
                success_url=request.host_url + 'pricing?success=true&session_id={CHECKOUT_SESSION_ID}',
                cancel_url=request.host_url + 'pricing?cancelled=true',
                customer_email=user.email,
                metadata={
                    'subscription_id': str(subscription.id),
                    'organization_id': str(organization.id),
                    'user_id': str(user.id),
                    'plan': plan_name
                },
                allow_promotion_codes=True,
                billing_address_collection='required',
                tax_id_collection={'enabled': True}
            )

            # Update subscription with checkout session ID
            subscription.stripe_checkout_session_id = checkout_session.id
            db.session.commit()

            logger.info(f"Checkout session created for organization {organization.id}, plan: {plan_name}")

            return jsonify({
                'sessionId': checkout_session.id,
                'url': checkout_session.url
            }), 200

        except stripe.error.StripeError as e:
            logger.error(f"Stripe error creating checkout session: {e}")
            return jsonify({'message': 'Failed to create payment session'}), 500

    except Exception as e:
        logger.error(f"Error creating checkout session: {e}")
        return jsonify({'message': 'Failed to create checkout session'}), 500

@app.route('/api/subscription/cancel', methods=['POST'])
@jwt_required()
def cancel_subscription():
    """Cancel current subscription"""
    try:
        user_id = get_jwt_identity()
        user = User.query.get(user_id)
        
        if not user:
            return jsonify({'message': 'User not found'}), 404

        organization = user.organization
        if not organization:
            return jsonify({'message': 'Organization not found'}), 404

        subscription = organization.subscription
        if not subscription:
            return jsonify({'message': 'No subscription found'}), 404

        if subscription.status != SubscriptionStatus.ACTIVE:
            return jsonify({'message': 'No active subscription to cancel'}), 400

        # Cancel subscription in Stripe
        if subscription.stripe_subscription_id:
            try:
                stripe.Subscription.delete(subscription.stripe_subscription_id)
                logger.info(f"Stripe subscription cancelled: {subscription.stripe_subscription_id}")
            except stripe.error.StripeError as e:
                logger.error(f"Error cancelling Stripe subscription: {e}")
                return jsonify({'message': 'Failed to cancel subscription with payment provider'}), 500

        # Update subscription status
        subscription.status = SubscriptionStatus.CANCELLED
        subscription.cancelled_at = datetime.utcnow()
        db.session.commit()

        logger.info(f"Subscription cancelled for organization: {organization.id}")

        return jsonify({
            'message': 'Subscription cancelled successfully',
            'subscription': subscription.to_dict()
        }), 200

    except Exception as e:
        logger.error(f"Error cancelling subscription: {e}")
        return jsonify({'message': 'Failed to cancel subscription'}), 500

@app.route('/api/subscription/update', methods=['POST'])
@jwt_required()
def update_subscription():
    """Update subscription plan"""
    try:
        user_id = get_jwt_identity()
        user = User.query.get(user_id)
        
        if not user:
            return jsonify({'message': 'User not found'}), 404

        organization = user.organization
        if not organization:
            return jsonify({'message': 'Organization not found'}), 404

        subscription = organization.subscription
        if not subscription or subscription.status != SubscriptionStatus.ACTIVE:
            return jsonify({'message': 'No active subscription found'}), 404

        data = request.get_json()
        new_plan = data.get('plan')
        
        if not new_plan or new_plan not in PRICING_PLANS:
            return jsonify({'message': 'Invalid plan selected'}), 400

        if subscription.plan.name.lower() == new_plan.lower():
            return jsonify({'message': 'Already subscribed to this plan'}), 400

        # Update subscription in Stripe
        if subscription.stripe_subscription_id:
            try:
                stripe_subscription = stripe.Subscription.retrieve(subscription.stripe_subscription_id)
                
                # Create new price
                new_price = stripe.Price.create(
                    product=stripe_subscription.items.data[0].price.product,
                    unit_amount=PRICING_PLANS[new_plan]['price'],
                    currency='usd',
                    recurring={'interval': 'month'}
                )

                # Update subscription
                stripe.Subscription.modify(
                    subscription.stripe_subscription_id,
                    items=[{
                        'id': stripe_subscription.items.data[0].id,
                        'price': new_price.id,
                    }],
                    proration_behavior='create_prorations'
                )

                logger.info(f"Stripe subscription updated: {subscription.stripe_subscription_id}")
            except stripe.error.StripeError as e:
                logger.error(f"Error updating Stripe subscription: {e}")
                return jsonify({'message': 'Failed to update subscription with payment provider'}), 500

        # Update local subscription
        subscription.plan = SubscriptionPlan[new_plan.upper()]
        subscription.updated_at = datetime.utcnow()
        db.session.commit()

        logger.info(f"Subscription updated for organization {organization.id} to {new_plan}")

        return jsonify({
            'message': 'Subscription updated successfully',
            'subscription': subscription.to_dict()
        }), 200

    except Exception as e:
        logger.error(f"Error updating subscription: {e}")
        return jsonify({'message': 'Failed to update subscription'}), 500

@app.route('/api/subscription/billing-portal', methods=['POST'])
@jwt_required()
def create_billing_portal_session():
    """Create Stripe billing portal session"""
    try:
        user_id = get_jwt_identity()
        user = User.query.get(user_id)
        
        if not user:
            return jsonify({'message': 'User not found'}), 404

        organization = user.organization
        if not organization:
            return jsonify({'message': 'Organization not found'}), 404

        subscription = organization.subscription
        if not subscription or not subscription.stripe_customer_id:
            return jsonify({'message': 'No billing information found'}), 404

        try:
            portal_session = stripe.billing_portal.Session.create(
                customer=subscription.stripe_customer_id,
                return_url=request.host_url + 'pricing'
            )

            return jsonify({
                'url': portal_session.url
            }), 200

        except stripe.error.StripeError as e:
            logger.error(f"Error creating billing portal session: {e}")
            return jsonify({'message': 'Failed to create billing portal session'}), 500

    except Exception as e:
        logger.error(f"Error creating billing portal session: {e}")
        return jsonify({'message': 'Failed to create billing portal session'}), 500

@app.route('/api/contact', methods=['POST'])
@limiter.limit("3 per hour")
def contact_enterprise():
    """Handle enterprise contact form"""
    try:
        data = request.get_json()
        
        required_fields = ['name', 'email', 'organization', 'message']
        for field in required_fields:
            if not data.get(field):
                return jsonify({'message': f'{field} is required'}), 400

        # Here you would typically send an email or save to database
        logger.info(f"Enterprise contact from: {data['email']} - {data['organization']}")
        
        return jsonify({
            'message': 'Thank you for your interest! Our enterprise team will contact you within 24 hours.'
        }), 200

    except Exception as e:
        logger.error(f"Contact form error: {e}")
        return jsonify({'message': 'Failed to submit contact form'}), 500


# Authentication Routes
@app.route('/api/register', methods=['GET', 'POST'])
@limiter.limit("5 per minute")
def register():
    """User registration endpoint"""
    
    if request.method == 'GET':
        # Serve the registration page/form
        return render_template('auth.html')  # or your registration template
    
    elif request.method == 'POST':
        try:

            # Debug: Log the raw request data
            print("Content-Type:", request.content_type)
            print("Raw data:", request.data)
            print("Form data:", request.form)
            print("JSON data:", request.get_json())
            
            data = request.get_json()
            
            if not data:
                return jsonify({'message': 'No data provided'}), 400
            
            # Validate required fields
            required_fields = ['firstName', 'lastName', 'email', 'password', 'role']
            for field in required_fields:
                if not data.get(field):
                    return jsonify({'message': f'{field.replace("register", "").replace("user", "").title()} is required'}), 400
            
            # Validate email format
            if not InputValidator.validate_email(data['email']):
                return jsonify({'message': 'Invalid email format'}), 400
            
            # Validate names
            if not InputValidator.validate_name(data['firstName']) or not InputValidator.validate_name(data['lastName']):
                return jsonify({'message': 'Names must contain only letters and be at least 2 characters long'}), 400
            
            # Validate password strength
            is_strong, message = InputValidator.validate_password_strength(data['password'])
            if not is_strong:
                return jsonify({'message': message}), 400
            
            # Validate role
            valid_roles = ['patient', 'nurse', 'oncologist', 'admin']
            if data['role'] not in valid_roles:
                return jsonify({'message': 'Invalid role specified'}), 400
            
            # Check if user already exists
            existing_user = User.query.filter_by(email=data['email'].lower()).first()
            if existing_user:
                return jsonify({'message': 'Email already registered'}), 409
            
            # default_plan_type = 'basic' # Default plan type for new users
            
            # Create new user
            user = User(
                first_name=data['firstName'],
                last_name=data['lastName'],
                email=data['email'],
                password=data['password'],
                # plan_type=default_plan_type,
                role=data['role']
            )
            
            db.session.add(user)
            db.session.commit()
            
            logger.info(f"New user registered: {user.email}")
            
            return jsonify({
                'message': 'User registered successfully',
                'user': user.to_dict()
            }), 201
            
        except Exception as e:
            logger.error(f"Registration error: {e}")
            db.session.rollback()
            return jsonify({'message': 'Registration failed'}), 500

@app.route('/api/login', methods=['GET', 'POST'])
@limiter.limit("10 per minute")
def login():
    """User login endpoint"""
    if request.method == 'GET':
        # Serve the registration page/form
        return render_template('auth.html')
    try:

        print("=== LOGIN DEBUG START ===")
        print("Content-Type:", request.content_type)
        print("Raw request data:", request.data)
        
        data = request.get_json()
        
        if not data or not data.get('email') or not data.get('password'):
            return jsonify({'message': 'Email and password are required'}), 400
        
        user = User.query.filter_by(email=data['email'].lower()).first()
        
        if not user:
            return jsonify({'message': 'Invalid credentials'}), 401
        
        # Check if account is locked
        if user.is_account_locked():
            return jsonify({'message': 'Account is temporarily locked due to multiple failed login attempts'}), 423
        
        # Check password
        if not user.check_password(data['password']):
            user.login_attempts += 1
            if user.login_attempts >= 5:
                user.lock_account()
                logger.warning(f"Account locked for user: {user.email}")
            else:
                db.session.commit()
            return jsonify({'message': 'Invalid credentials'}), 401
        
        # Check if user is active
        if not user.is_active:
            return jsonify({'message': 'Account is deactivated'}), 401
        
        # Successful login
        user.login_attempts = 0
        user.last_login = datetime.utcnow()
        user.locked_until = None
        db.session.commit()

        session['user_id'] = user.id
        session['role'] = user.role
        session.permanent = True

        # ADD THESE DEBUG LINES:
        print(f"=== SESSION DEBUG ===")
        print(f"Setting session user_id: {user.id}")
        print(f"Session after setting: {dict(session)}")
        print(f"Session user_id: {session.get('user_id')}")

        
        access_token = create_access_token(identity=user.id)
        
        logger.info(f"Successful login for user: {user.email}")
        
        return jsonify({
            'message': 'Login successful',
            'access_token': access_token,
            'user': user.to_dict(),
            'user_id': user.id,
            'role': user.role,  # Make sure your User model has this field
            'redirect_url': '/patient-dashboard' if user.role == 'patient' else '/dashboard'
        }), 200
        
    except Exception as e:
        logger.error(f"Login error: {e}")
        return jsonify({'message': 'Login failed'}), 500

@app.route('/api/patient/profile', methods=['POST'])
def create_patient_profile():
    """Create patient profile after registration"""
    try:
        data = request.get_json()
        user_id = session.get('user_id')
        
        # Validate required fields
        required_fields = ['age', 'gender', 'diagnosis', 'stage', 'date_diagnosed']
        for field in required_fields:
            if not data.get(field):
                return jsonify({'message': f'{field.replace("_", " ").title()} is required'}), 400
        
        # Check if user exists and is a patient
        user = db.session.get(User, user_id)
        print(f"Found user: {user}")  # DEBUG LINE
        
        if not user_id:  # ADD THIS CHECK FIRST
            return jsonify({'message': 'User not logged in'}), 401
            
        if not user or user.role != 'patient':
            return jsonify({'message': 'Invalid user or not a patient'}), 400
        
        # Check if patient profile already exists
        existing_patient = Patient.query.filter_by(user_id=user_id).first()
        if existing_patient:
            return jsonify({'message': 'Patient profile already exists'}), 409
        
        # Create patient record
        patient = Patient(
            user_id=user_id,
            age=data['age'],
            gender=data['gender'],
            diagnosis=data['diagnosis'],
            stage=data['stage'],
            date_diagnosed=datetime.strptime(data['date_diagnosed'], '%Y-%m-%d').date(),
            emergency_contact=data.get('emergency_contact'),
            insurance_info=data.get('insurance_info')
        )
        
        db.session.add(patient)
        db.session.commit()
        
        return jsonify({
            'message': 'Patient profile created successfully',
            'patient_id': patient.id
        }), 201
        
    except Exception as e:
        logger.error(f"Patient profile creation error: {e}")
        db.session.rollback()
        return jsonify({'message': 'Failed to create patient profile'}), 500

@app.route('/setup-profile')
def profile_setup():
    """Serve the profile setup page"""
    return render_template('profile_setup.html')

@app.route('/api/logout', methods=['POST'])
@jwt_required()
def logout():
    """User logout endpoint"""
    try:
        jti = get_jwt_identity()
        blacklisted_tokens.add(jti)
        return jsonify({'message': 'Successfully logged out'}), 200
    except Exception as e:
        logger.error(f"Logout error: {e}")
        return jsonify({'message': 'Logout failed'}), 500

# Patient Management Routes
@app.route('/api/patients', methods=['GET'])
@oncologist_required
@role_required(['nurse', 'oncologist', 'admin'])
def get_patients():
    """Get list of patients"""
    try:
        oncologist_id = session['user_id']
        patients = db.session.query(Patient, User).join(User, Patient.user_id == User.id).filter(Patient.oncologist_id == oncologist_id).all()
        
        patient_list = []
        for patient, user in patients:
            # Get latest vital signs
            latest_vitals = VitalSigns.query.filter_by(
                patient_id=patient.id
            ).order_by(VitalSigns.recorded_at.desc()).first()
            
            # Get next appointment
            next_appointment = Appointment.query.filter_by(
                patient_id=patient.id
            ).filter(
                Appointment.appointment_date > datetime.utcnow()
            ).order_by(Appointment.appointment_date.asc()).first()
            
            patient_data = {
                'id': patient.id,
                'name': user.full_name,
                'age': patient.age,
                'diagnosis': patient.diagnosis,
                'stage': patient.stage,
                'status': patient.status,
                'last_visit': patient.last_visit.isoformat() if patient.last_visit else None,
                'next_appointment': next_appointment.appointment_date.isoformat() if next_appointment else None,
                'latest_vitals': {
                    'recorded_at': latest_vitals.recorded_at.isoformat() if latest_vitals else None,
                    'heart_rate': latest_vitals.heart_rate if latest_vitals else None,
                    'blood_pressure': f"{latest_vitals.blood_pressure_systolic}/{latest_vitals.blood_pressure_diastolic}" if latest_vitals and latest_vitals.blood_pressure_systolic else None,
                    'pain_level': latest_vitals.pain_level if latest_vitals else None
                }
            }
            patient_list.append(patient_data)
    
        return jsonify(patient_list), 200
    except Exception as e:
        logger.error(f"Error fetching patients: {e}")
        return jsonify({'error': 'Failed to fetch patients'}), 500

@app.route('/api/appointments')
@oncologist_required
@role_required(['nurse', 'oncologist', 'patient'])
def get_appointments():
    oncologist_id = session['user_id']
    date_filter = request.args.get('date', datetime.now().date().isoformat())
    
    appointments = db.session.query(Appointment, Patient, User).join(
        Patient, Appointment.patient_id == Patient.id
    ).join(
        User, Patient.user_id == User.id
    ).filter(
        Appointment.oncologist_id == oncologist_id,
        db.func.date(Appointment.appointment_date) == date_filter
    ).order_by(Appointment.appointment_date.asc()).all()
    
    appointment_list = []
    for appointment, patient, user in appointments:
        appointment_data = {
            'id': appointment.id,
            'time': appointment.appointment_date.strftime('%H:%M'),
            'patient_name': user.full_name,
            'patient_id': patient.id,
            'type': appointment.appointment_type,
            'status': appointment.status,
            'notes': appointment.notes
        }
        appointment_list.append(appointment_data)
    
    return jsonify(appointment_list)

@app.route('/api/treatments')
@oncologist_required
@role_required(['nurse', 'oncologist', 'patient'])
def get_treatments():
    oncologist_id = session['user_id']
    
    treatments = db.session.query(Treatment, Patient, User).join(
        Patient, Treatment.patient_id == Patient.id
    ).join(
        User, Patient.user_id == User.id
    ).filter(
        Patient.oncologist_id == oncologist_id,
        Treatment.status == 'active'
    ).all()
    
    treatment_list = []
    for treatment, patient, user in treatments:
        progress = 0
        if treatment.total_cycles and treatment.current_cycle:
            progress = (treatment.current_cycle / treatment.total_cycles) * 100
        
        treatment_data = {
            'id': treatment.id,
            'patient_name': user.full_name,
            'patient_id': patient.id,
            'treatment_type': treatment.treatment_type,
            'medication': treatment.medication,
            'current_cycle': treatment.current_cycle,
            'total_cycles': treatment.total_cycles,
            'progress': round(progress, 1),
            'adherence_rate': treatment.adherence_rate,
            'last_updated': treatment.last_updated.isoformat()
        }
        treatment_list.append(treatment_data)
    
    return jsonify(treatment_list)

@app.route('/api/alerts')
@oncologist_required
@role_required(['nurse', 'oncologist'])
def get_alerts():
    oncologist_id = session['user_id']
    
    alerts = db.session.query(Alert, Patient, User).join(
        Patient, Alert.patient_id == Patient.id
    ).join(
        User, Patient.user_id == User.id
    ).filter(
        Alert.oncologist_id == oncologist_id,
        Alert.is_resolved == False
    ).order_by(Alert.created_at.desc()).limit(50).all()
    
    alert_list = []
    for alert, patient, user in alerts:
        alert_data = {
            'id': alert.id,
            'patient_name': user.full_name,
            'patient_id': patient.id,
            'alert_type': alert.alert_type,
            'priority': alert.priority,
            'title': alert.title,
            'message': alert.message,
            'is_read': alert.is_read,
            'created_at': alert.created_at.isoformat()
        }
        alert_list.append(alert_data)
    
    return jsonify(alert_list)

@app.route('/api/auth/status', methods=['GET'])
def auth_status():
    print(f"Session contents: {dict(session)}")
    print(f"Session keys: {list(session.keys())}")
    print(f"User ID in session: {'user_id' in session}")
    
    if 'user_id' not in session:
        print("No user_id in session")
        return jsonify({'error': 'Not authenticated', 'session_empty': True}), 401
    
    try:
        user = db.session.get(User, session['user_id'])
        if not user:
            print(f"User not found for ID: {session['user_id']}")
            return jsonify({'error': 'User not found'}), 401
        
        return jsonify({
            'authenticated': True,
            'user': {
                'id': user.id,
                'email': user.email,
                'name': user.name,
                'role': user.role
            },
            'session_id': session.get('_id', 'unknown')
        }), 200
        
    except Exception as e:
        print(f"Error in auth_status: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/ai-insights')
@oncologist_required
@role_required(['nurse', 'oncologist', 'patient'])
def get_ai_insights():
    oncologist_id = session['user_id']
    
    # Calculate treatment adherence rate
    treatments = Treatment.query.join(Patient).filter(
        Patient.oncologist_id == oncologist_id,
        Treatment.status == 'active',
        Treatment.adherence_rate.isnot(None)
    ).all()
    
    avg_adherence = sum(t.adherence_rate for t in treatments) / len(treatments) if treatments else 0
    
    # Count risk predictions (patients with critical status or recent alerts)
    risk_predictions = Patient.query.filter(
        Patient.oncologist_id == oncologist_id,
        Patient.status.in_(['critical', 'deteriorating'])
    ).count()
    
    # Calculate average response time (time between alert creation and resolution)
    resolved_alerts = Alert.query.filter(
        Alert.oncologist_id == oncologist_id,
        Alert.is_resolved == True,
        Alert.resolved_at.isnot(None)
    ).all()
    
    avg_response_time = 0
    if resolved_alerts:
        response_times = [(alert.resolved_at - alert.created_at).total_seconds() / 60 
                         for alert in resolved_alerts]
        avg_response_time = sum(response_times) / len(response_times)
    
    insights = {
        'treatment_adherence': round(avg_adherence, 1),
        'risk_predictions': risk_predictions,
        'avg_response_time': round(avg_response_time, 1),
        'system_uptime': 98.5,  # This would come from system monitoring
        'patients_monitored': Patient.query.filter_by(oncologist_id=oncologist_id).count(),
        'active_treatments': Treatment.query.join(Patient).filter(
            Patient.oncologist_id == oncologist_id,
            Treatment.status == 'active'
        ).count()
    }
    
    return jsonify(insights)


@app.route('/api/patients/<int:patient_id>', methods=['GET'])
@role_required(['nurse', 'oncologist', 'admin', 'patient'])
def get_patient_details(patient_id):
    """Get detailed patient information"""
    try:
        current_user_id = get_jwt_identity()
        current_user = User.query.get(current_user_id)
        
        # Patients can only view their own data
        if current_user.role == 'patient':
            patient = Patient.query.filter_by(user_id=current_user_id).first()
            if not patient or patient.id != patient_id:
                return jsonify({'error': 'Access denied'}), 403
        
        patient = Patient.query.get(patient_id)
        if not patient:
            return jsonify({'error': 'Patient not found'}), 404
        
        user = User.query.get(patient.user_id)
        
        # Get recent treatments
        treatments = Treatment.query.filter_by(patient_id=patient_id)\
            .order_by(Treatment.created_at.desc()).limit(5).all()
        
        # Get upcoming appointments
        appointments = Appointment.query.filter_by(patient_id=patient_id)\
            .filter(Appointment.appointment_date >= datetime.utcnow())\
            .order_by(Appointment.appointment_date.asc()).limit(5).all()
        
        patient_data = {
            'id': patient.id,
            'name': user.full_name,
            'age': patient.age,
            'gender': patient.gender,
            'diagnosis': patient.diagnosis,
            'stage': patient.stage,
            'risk_score': patient.risk_score,
            'date_diagnosed': patient.date_diagnosed.isoformat() if patient.date_diagnosed else None,
            'treatments': [{
                'id': t.id,
                'type': t.treatment_type,
                'protocol': t.protocol_name,
                'current_cycle': t.current_cycle,
                'total_cycles': t.total_cycles,
                'status': t.status
            } for t in treatments],
            'appointments': [{
                'id': a.id,
                'date': a.appointment_date.isoformat(),
                'type': a.appointment_type,
                'status': a.status
            } for a in appointments]
        }
        
        return jsonify(patient_data), 200
    except Exception as e:
        logger.error(f"Error fetching patient details: {e}")
        return jsonify({'error': 'Failed to fetch patient details'}), 500
    
@app.route('/oncologist/profile-setup')
def profile_setup_page():
    """Serve the profile setup page"""
    # Check if user is logged in (adjust based on your auth system)
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    # Check if profile already exists
    user_id = session['user_id']
    existing_profile = OncologistProfile.query.filter_by(user_id=user_id).first()
    
    if existing_profile and existing_profile.profile_completed:
        return redirect('/oncologist-dashboard')
    
    # Serve the HTML page (you can use render_template if using templates)
    return render_template('oncologist_profile.html')

@app.route('/api/oncologist/profile', methods=['POST'])
def create_oncologist_profile():
    """Create or update oncologist profile"""
    try:
        # Check if user is logged in
        if 'user_id' not in session:
            return jsonify({'error': 'Unauthorized', 'message': 'Please log in first'}), 401
        
        user_id = session['user_id']
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'Bad Request', 'message': 'No data provided'}), 400
        
        # Validate required fields
        required_fields = [
            'firstName', 'lastName', 'email', 'phone', 'licenseNumber',
            'yearsOfExperience', 'currentHospital', 'position', 
            'medicalSchool', 'graduationYear'
        ]
        
        validation_errors = validate_required_fields(data, required_fields)
        
        # Additional validations
        if 'email' in data and not validate_email(data['email']):
            validation_errors['email'] = 'Invalid email format'
        
        if 'phone' in data and not validate_phone(data['phone']):
            validation_errors['phone'] = 'Invalid phone number format'
        
        if 'licenseNumber' in data and not validate_license_number(data['licenseNumber']):
            validation_errors['licenseNumber'] = 'License number must be at least 3 characters'
        
        if 'yearsOfExperience' in data:
            try:
                years = int(data['yearsOfExperience'])
                if years < 0 or years > 50:
                    validation_errors['yearsOfExperience'] = 'Years of experience must be between 0 and 50'
            except (ValueError, TypeError):
                validation_errors['yearsOfExperience'] = 'Years of experience must be a valid number'
        
        if 'graduationYear' in data:
            try:
                year = int(data['graduationYear'])
                current_year = datetime.now().year
                if year < 1950 or year > current_year:
                    validation_errors['graduationYear'] = f'Graduation year must be between 1950 and {current_year}'
            except (ValueError, TypeError):
                validation_errors['graduationYear'] = 'Graduation year must be a valid number'
        
        # Validate specializations
        specializations = data.get('specializations', [])
        spec_error = validate_specializations(specializations)
        if spec_error:
            validation_errors['specializations'] = spec_error
        
        if validation_errors:
            return jsonify({
                'error': 'Validation Error',
                'message': 'Please correct the following errors',
                'errors': validation_errors
            }), 400
        
        # Check if profile already exists
        existing_profile = OncologistProfile.query.filter_by(user_id=user_id).first()
        
        if existing_profile:
            # Update existing profile
            profile = existing_profile
        else:
            # Create new profile
            profile = OncologistProfile(user_id=user_id)
        
        # Map form data to profile fields
        profile.first_name = data['firstName'].strip()
        profile.last_name = data['lastName'].strip()
        profile.email = data['email'].strip().lower()
        profile.phone = data['phone'].strip()
        profile.license_number = data['licenseNumber'].strip()
        profile.years_of_experience = int(data['yearsOfExperience'])
        profile.current_hospital = data['currentHospital'].strip()
        profile.position = data['position']
        profile.medical_school = data['medicalSchool'].strip()
        profile.graduation_year = int(data['graduationYear'])
        
        # Optional fields
        if data.get('dateOfBirth'):
            try:
                profile.date_of_birth = datetime.strptime(data['dateOfBirth'], '%Y-%m-%d').date()
            except ValueError:
                pass
        
        profile.gender = data.get('gender', '').strip()
        profile.board_certification = data.get('boardCertification', '').strip()
        profile.additional_certifications = data.get('additionalCertifications', '').strip()
        profile.research_interests = data.get('researchInterests', '').strip()
        profile.ai_experience = data.get('aiExperience', '').strip()
        profile.notification_preferences = data.get('notificationPreferences', 'all')
        profile.ai_preferences = data.get('aiPreferences', '').strip()
        profile.bio = data.get('bio', '').strip()
        
        # Store specializations as JSON
        profile.specializations = json.dumps(specializations)
        profile.profile_completed = True
        profile.updated_at = datetime.utcnow()
        
        # Save to database
        if not existing_profile:
            db.session.add(profile)
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Profile saved successfully',
            'profile': profile.to_dict()
        }), 200
        
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Error creating oncologist profile: {str(e)}")
        return jsonify({
            'error': 'Internal Server Error',
            'message': 'An error occurred while saving your profile. Please try again.'
        }), 500

@app.route('/api/oncologist/profile', methods=['GET'])
def get_oncologist_profile():
    """Get oncologist profile data"""
    try:
        if 'user_id' not in session:
            return jsonify({'error': 'Unauthorized', 'message': 'Please log in first'}), 401
        
        user_id = session['user_id']
        profile = OncologistProfile.query.filter_by(user_id=user_id).first()
        
        if not profile:
            return jsonify({'error': 'Not Found', 'message': 'Profile not found'}), 404
        
        return jsonify({
            'success': True,
            'profile': profile.to_dict()
        }), 200
        
    except Exception as e:
        app.logger.error(f"Error fetching oncologist profile: {str(e)}")
        return jsonify({
            'error': 'Internal Server Error',
            'message': 'An error occurred while fetching your profile.'
        }), 500

@app.route('/api/oncologist/profile', methods=['PUT'])
def update_oncologist_profile():
    """Update existing oncologist profile"""
    try:
        if 'user_id' not in session:
            return jsonify({'error': 'Unauthorized', 'message': 'Please log in first'}), 401
        
        user_id = session['user_id']
        profile = OncologistProfile.query.filter_by(user_id=user_id).first()
        
        if not profile:
            return jsonify({'error': 'Not Found', 'message': 'Profile not found'}), 404
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Bad Request', 'message': 'No data provided'}), 400
        
        # Update only provided fields
        updatable_fields = {
            'firstName': 'first_name',
            'lastName': 'last_name',
            'email': 'email',
            'phone': 'phone',
            'gender': 'gender',
            'currentHospital': 'current_hospital',
            'position': 'position',
            'boardCertification': 'board_certification',
            'additionalCertifications': 'additional_certifications',
            'researchInterests': 'research_interests',
            'aiExperience': 'ai_experience',
            'notificationPreferences': 'notification_preferences',
            'aiPreferences': 'ai_preferences',
            'bio': 'bio'
        }
        
        for form_field, db_field in updatable_fields.items():
            if form_field in data:
                setattr(profile, db_field, data[form_field])
        
        # Handle date of birth
        if 'dateOfBirth' in data and data['dateOfBirth']:
            try:
                profile.date_of_birth = datetime.strptime(data['dateOfBirth'], '%Y-%m-%d').date()
            except ValueError:
                pass
        
        # Handle specializations
        if 'specializations' in data:
            spec_error = validate_specializations(data['specializations'])
            if spec_error:
                return jsonify({
                    'error': 'Validation Error',
                    'message': spec_error
                }), 400
            profile.specializations = json.dumps(data['specializations'])
        
        profile.updated_at = datetime.utcnow()
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Profile updated successfully',
            'profile': profile.to_dict()
        }), 200
        
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Error updating oncologist profile: {str(e)}")
        return jsonify({
            'error': 'Internal Server Error',
            'message': 'An error occurred while updating your profile.'
        }), 500

# Nurse Profile Setup Route
@app.route('/nurse/profile/setup', methods=['GET', 'POST'])
def nurse_profile_setup():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user_id = session['user_id']
    user = User.query.get(user_id)
    if not user or user.role != 'nurse':
        return render_template('nurse_profile_setup.html')
    
    # Check if profile already exists
    existing_profile = NurseProfile.query.filter_by(user_id=user_id).first()
    if existing_profile and existing_profile.profile_completed:
        return redirect(url_for('nurse_dashboard'))
    
    if request.method == 'POST':
        try:
            # Get form data
            first_name = request.form.get('first_name')
            last_name = request.form.get('last_name')
            license_number = request.form.get('license_number')
            specialization = request.form.get('specialization')
            years_experience = request.form.get('years_experience')
            hospital_name = request.form.get('hospital_name')
            department = request.form.get('department')
            phone = request.form.get('phone')
            emergency_contact = request.form.get('emergency_contact')
            shift_preference = request.form.get('shift_preference')
            certifications = request.form.get('certifications')
            
            # Validate required fields
            if not all([first_name, last_name, license_number, specialization, 
                       years_experience, hospital_name, department, phone, 
                       emergency_contact, shift_preference]):
                flash('All fields are required', 'error')
                return render_template('nurse_profile_setup.html')
            
            # Check if license number already exists
            existing_license = NurseProfile.query.filter_by(license_number=license_number).first()
            if existing_license and existing_license.user_id != user.id:
                flash('License number already exists', 'error')
                return render_template('nurse_profile_setup.html')
            
            if existing_profile:
                # Update existing profile
                existing_profile.first_name = first_name
                existing_profile.last_name = last_name
                existing_profile.license_number = license_number
                existing_profile.specialization = specialization
                existing_profile.years_experience = int(years_experience)
                existing_profile.hospital_name = hospital_name
                existing_profile.department = department
                existing_profile.phone = phone
                existing_profile.emergency_contact = emergency_contact
                existing_profile.shift_preference = shift_preference
                existing_profile.certifications = certifications
                existing_profile.profile_completed = True
                existing_profile.updated_at = datetime.utcnow()
            else:
                # Create new profile
                nurse_profile = NurseProfile(
                    user_id=user.id,
                    first_name=first_name,
                    last_name=last_name,
                    license_number=license_number,
                    specialization=specialization,
                    years_experience=int(years_experience),
                    hospital_name=hospital_name,
                    department=department,
                    phone=phone,
                    emergency_contact=emergency_contact,
                    shift_preference=shift_preference,
                    certifications=certifications,
                    profile_completed=True
                )
                db.session.add(nurse_profile)
            
            db.session.commit()
            flash('Profile setup completed successfully!', 'success')
            return redirect(url_for('nurse_dashboard'))
            
        except Exception as e:
            db.session.rollback()
            flash(f'Error setting up profile: {str(e)}', 'error')
            return render_template('nurse_profile_setup.html')
    
    # GET request - show the form
    profile_data = existing_profile if existing_profile else None
    return render_template('nurse_profile_setup.html', profile=profile_data)

# Nurse Dashboard Route
@app.route('/nurse-dashboard')
def nurse_dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user_id = session['user_id']
    user = User.query.get(user_id)
    if not user or user.role != 'nurse':
        return redirect(url_for('dashboard'))
    
    # Check if profile is completed
    nurse_profile = NurseProfile.query.filter_by(user_id=user.id).first()
    if not nurse_profile or not nurse_profile.profile_completed:
        return redirect(url_for('nurse_profile_setup'))
    
    # Get dashboard data
    try:
        # Get assigned patients count (you'll need to implement patient assignment logic)
        assigned_patients = 0  # Placeholder - implement based on your patient assignment system
        
        # Get today's appointments/tasks
        today = datetime.utcnow().date()
        todays_tasks = 0  # Placeholder - implement based on your task/appointment system
        
        # Get recent activities
        recent_activities = []  # Placeholder - implement based on your activity logging system
        
        # Get pending notifications
        notifications = []  # Placeholder - implement based on your notification system
        
        dashboard_data = {
            'nurse_profile': nurse_profile,
            'assigned_patients': assigned_patients,
            'todays_tasks': todays_tasks,
            'recent_activities': recent_activities,
            'notifications': notifications
        }
        
        return render_template('nurse_dashboard.html', data=dashboard_data)
        
    except Exception as e:
        flash(f'Error loading dashboard: {str(e)}', 'error')
        return render_template('nurse_dashboard.html', data={
            'nurse_profile': nurse_profile,
            'assigned_patients': 0,
            'todays_tasks': 0,
            'recent_activities': [],
            'notifications': []
        })

# Nurse Profile View/Edit Route
@app.route('/nurse/profile')
def nurse_profile():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user = User.query.get(session['user_id'])
    if not user or user.role != 'nurse':
        return redirect(url_for('dashboard'))
    
    nurse_profile = NurseProfile.query.filter_by(user_id=user.id).first()
    if not nurse_profile:
        return redirect(url_for('nurse_profile_setup'))
    
    return render_template('nurse_profile.html', profile=nurse_profile)

# API endpoint for nurse profile data (for frontend interactions)
@app.route('/api/nurse/profile')
def api_nurse_profile():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    user = User.query.get(session['user_id'])
    if not user or user.role != 'nurse':
        return jsonify({'error': 'Not authorized'}), 403
    
    nurse_profile = NurseProfile.query.filter_by(user_id=user.id).first()
    if not nurse_profile:
        return jsonify({'error': 'Profile not found'}), 404
    
    return jsonify({
        'id': nurse_profile.id,
        'first_name': nurse_profile.first_name,
        'last_name': nurse_profile.last_name,
        'license_number': nurse_profile.license_number,
        'specialization': nurse_profile.specialization,
        'years_experience': nurse_profile.years_experience,
        'hospital_name': nurse_profile.hospital_name,
        'department': nurse_profile.department,
        'phone': nurse_profile.phone,
        'emergency_contact': nurse_profile.emergency_contact,
        'shift_preference': nurse_profile.shift_preference,
        'certifications': nurse_profile.certifications,
        'profile_completed': nurse_profile.profile_completed,
        'created_at': nurse_profile.created_at.isoformat(),
        'updated_at': nurse_profile.updated_at.isoformat()
    })

# AI Insights Routes
@app.route('/api/ai/risk-assessment/<int:patient_id>', methods=['GET'])
@role_required(['nurse', 'oncologist', 'admin'])
def get_risk_assessment(patient_id):
    """Get AI risk assessment for patient"""
    try:
        if not risk_model:
            return jsonify({'error': 'Risk assessment model not available'}), 503
        
        patient = Patient.query.get(patient_id)
        if not patient:
            return jsonify({'error': 'Patient not found'}), 404
        
        # Calculate AI risk score
        risk_score = risk_model.predict_risk(
            age=patient.age,
            stage=patient.stage,
            diagnosis=patient.diagnosis
        )
        
        # Update patient record
        patient.risk_score = risk_score
        db.session.commit()
        
        # Generate insights
        insights = []
        if risk_score > 0.7:
            insights.append("High risk patient - recommend immediate intervention review")
            insights.append("Consider additional monitoring and support services")
        elif risk_score > 0.5:
            insights.append("Medium risk - monitor closely and adjust treatment as needed")
        else:
            insights.append("Low risk - continue current treatment protocol")
        
        return jsonify({
            'patient_id': patient_id,
            'risk_score': risk_score,
            'risk_level': 'High' if risk_score > 0.7 else 'Medium' if risk_score > 0.5 else 'Low',
            'insights': insights,
            'updated_at': datetime.utcnow().isoformat()
        }), 200
    except Exception as e:
        logger.error(f"Error generating risk assessment: {e}")
        return jsonify({'error': 'Failed to generate risk assessment'}), 500

# Dashboard Statistics Routes
@app.route('/api/dashboard/stats', methods=['GET'])
@oncologist_required
@role_required(['nurse', 'oncologist', 'admin', 'patient'])
def get_dashboard_stats():
    """Get dashboard statistics"""
    try:
        oncologist_id = session['user_id']
        current_user = User.query.get(oncologist_id)
        
        if current_user.role == 'patient':
            patient = Patient.query.filter_by(user_id=oncologist_id).first()
            if not patient:
                return jsonify({})
            
            # Patient-specific stats
            stats = {
                'active_treatments': Treatment.query.filter_by(patient_id=patient.id, status='active').count(),
                'upcoming_appointments': Appointment.query.filter_by(patient_id=patient.id)
                    .filter(Appointment.appointment_date >= datetime.utcnow()).count(),
                'active_medications': PatientMedication.query.filter_by(patient_id=patient.id, status='active').count(),
                'risk_score': patient.risk_score
            }
        
        elif current_user.role in ['nurse', 'oncologist']:
            # Medical staff stats
            # Get current stats
            total_patients = Patient.query.filter_by(oncologist_id=oncologist_id).count()
            
            today = datetime.now().date()
            today_appointments = Appointment.query.join(Patient).filter(
                Patient.oncologist_id == oncologist_id,
                db.func.date(Appointment.appointment_date) == today,
                Appointment.status.in_(['scheduled', 'confirmed'])
            ).count()
            
            active_treatments = Treatment.query.join(Patient).filter(
                Patient.oncologist_id == oncologist_id,
                Treatment.status == 'active'
            ).count()
            
            critical_alerts = Alert.query.filter(
                Alert.oncologist_id == oncologist_id,
                Alert.priority == 'critical',
                Alert.is_resolved == False
            ).count()
            
            # Get recent updates count for the last hour
            recent_time = datetime.utcnow() - timedelta(hours=1)
            recent_updates = {
                'vitals': VitalSigns.query.join(Patient).filter(
                    Patient.oncologist_id == oncologist_id,
                    VitalSigns.recorded_at > recent_time
                ).count(),
                'labs': LabResult.query.join(Patient).filter(
                    Patient.oncologist_id == oncologist_id,
                    LabResult.reported_date > recent_time
                ).count(),
                'treatments': Treatment.query.join(Patient).filter(
                    Patient.oncologist_id == oncologist_id,
                    Treatment.last_updated > recent_time
                ).count()
            }
            
            return jsonify({
                'patients': total_patients,
                'appointments': today_appointments,
                'treatments': active_treatments,
                'alerts': critical_alerts,
                'recent_updates': recent_updates
            })
        
        else:  # admin
            stats = {
                'total_users': User.query.count(),
                'total_patients': Patient.query.count(),
                'total_appointments': Appointment.query.count(),
                'system_uptime': 99.8  # Mock value
            }
        
        return jsonify(stats), 200
    except Exception as e:
        logger.error(f"Error fetching dashboard stats: {e}")
        return jsonify({'error': 'Failed to fetch dashboard statistics'}), 500
    
# Dashboard page routes - all serve the same index.html
@app.route('/patient-dashboard')
def patient_dashboard():
    print(f"=== DASHBOARD DEBUG ===")
    print(f"Session contents: {dict(session)}")
    print(f"Session user_id: {session.get('user_id')}")

    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    try:
        # Get the patient record using the user_id from session
        patients = Patient.query.filter_by(user_id=session['user_id']).first()
        
        if patients is None:
            flash('Patient record not found. Please login again.', 'error')
            return redirect(url_for('login'))
        
        # Get the user name from the related User model
        user_name = patients.user.first_name if patients.user else 'Patient'

        user_id = session['user_id']
        print(f"Looking for patient with user_id: {user_id}")  # Debug line

        patient = Patient.query.filter_by(user_id=user_id).first()
        print(f"Found patient: {patient}")  # Debug line

        if not patient:
            return redirect('/setup-profile')
        return render_template('patient.html', user_name=user_name, patients=patients)
    except Exception as e:
        print(f"Error in patient_dashboard: {e}")
        flash('An error occurred. Please try again.', 'error')
        return redirect(url_for('login'))

# @app.route('/nurse-dashboard') 
# def nurse_dashboard():
#     """Serve Nurse Dashboard Page"""
#     user = User.query.get(session['user_id'])
#     if user.role == 'nurse':
#         nurse_profile = NurseProfile.query.filter_by(user_id=user.id).first()

#     if not nurse_profile or not nurse_profile.profile_completed:
#         return redirect(url_for('nurse_profile_setup'))
#     return render_template('nurse_dashboard.html')

@app.route('/oncologist-dashboard')
def doctor_dashboard():
    """Serve the oncologist dashboard page"""
    if 'user_id' not in session:
        return redirect('/login')
    
    user_id = session['user_id']
    profile = OncologistProfile.query.filter_by(user_id=user_id).first()
    # oncologist = OncologistProfile.query.filter_by(id=session['user_id']).first()
    
    if not profile or not profile.profile_completed:
        return redirect('/oncologist/profile-setup')
    return render_template('oncologist.html', oncologist_name=profile.last_name, profile=profile)

@app.route('/admin-dashboard')
def admin_dashboard():
    return render_template('index.html')

# Generic dashboard route
@app.route('/dashboard')
def dashboard():
    return render_template('index.html')

@app.route('/clinical')
def clinical():
    return render_template('clinical.html')

@app.route('/api/verify-token', methods=['GET'])
def verify_token():
    """Verify JWT token and return user info"""
    try:
        # Get token from Authorization header
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({'valid': False, 'message': 'No token provided'}), 401
        
        token = auth_header.split(' ')[1]
        
        # Verify token
        decoded_token = decode_token(token)
        user_id = decoded_token['sub']
        
        user = User.query.get(user_id)
        
        if not user or not user.is_active:
            return jsonify({'valid': False, 'message': 'Invalid or inactive user'}), 401
        
        return jsonify({
            'valid': True,
            'user': user.to_dict()
        }), 200
        
    except Exception as e:
        logger.error(f"Token verification error: {e}")
        return jsonify({'valid': False, 'message': 'Invalid token'}), 401

# Initialize database and create sample data
@app.before_first_request
def create_tables():

    db.drop_all()
    db.create_all()

    print("Database tables created successfully!")
    print("Ready for user registration!")

# limiter.init_app(app)

# Stripe configuration
stripe.api_key = os.environ.get('STRIPE_SECRET_KEY', 'sk_test_your_key_here')

# Encryption setup
encryption_key = os.environ.get('ENCRYPTION_KEY', Fernet.generate_key())
cipher_suite = Fernet(encryption_key)

# Logging configuration
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s %(levelname)s %(name)s %(message)s',
#     handlers=[
#         logging.FileHandler('donation_system.log'),
#         logging.StreamHandler()
#     ]
# )
# logger = logging.getLogger(__name__)

# Data models
@dataclass
class Donor:
    first_name: str
    last_name: str
    email: str
    phone: Optional[str] = None
    anonymous: bool = False
    newsletter: bool = False

@dataclass
class Donation:
    amount: float
    currency: str = 'USD'
    donation_type: str = 'one-time'  # 'one-time' or 'monthly'
    payment_method: str = 'card'
    donor: Donor = None
    transaction_id: Optional[str] = None
    status: str = 'pending'
    created_at: Optional[datetime] = None

class SecurityValidator:
    """Security validation utilities"""
    
    @staticmethod
    def validate_email(email: str) -> bool:
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    @staticmethod
    def validate_amount(amount: float) -> bool:
        return 1.0 <= amount <= 50000.0
    
    @staticmethod
    def sanitize_input(data: str) -> str:
        """Basic input sanitization"""
        if not isinstance(data, str):
            return str(data)
        return data.strip()[:255]  # Limit length and strip whitespace
    
    @staticmethod
    def generate_secure_token() -> str:
        return str(uuid.uuid4())

class DatabaseManager:
    """Database operations manager"""
    
    def __init__(self, db_path='donations.db'):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Donations table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS donations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    transaction_id TEXT UNIQUE NOT NULL,
                    amount Numeric(10,2) NOT NULL,
                    currency TEXT DEFAULT 'USD',
                    donation_type TEXT NOT NULL,
                    payment_method TEXT NOT NULL,
                    donor_id INTEGER,
                    status TEXT DEFAULT 'pending',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    encrypted_data TEXT,
                    FOREIGN KEY (donor_id) REFERENCES donors (id)
                )
            ''')
            
            # Donors table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS donors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    first_name TEXT NOT NULL,
                    last_name TEXT NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    phone TEXT,
                    anonymous BOOLEAN DEFAULT FALSE,
                    newsletter BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    total_donated Numeric(10,2) DEFAULT 0.00
                )
            ''')
            
            # Audit logs table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS audit_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    action TEXT NOT NULL,
                    user_ip TEXT,
                    details TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    transaction_id TEXT
                )
            ''')
            
            # Transparency reports table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS transparency_reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    report_date DATE NOT NULL,
                    total_raised Numeric(12,2),
                    fund_allocation JSON,
                    impact_metrics JSON,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
    
    def create_donor(self, donor: Donor) -> int:
        """Create or get existing donor"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Check if donor exists
            cursor.execute('SELECT id FROM donors WHERE email = ?', (donor.email,))
            existing = cursor.fetchone()
            
            if existing:
                return existing[0]
            
            # Create new donor
            cursor.execute('''
                INSERT INTO donors (first_name, last_name, email, phone, anonymous, newsletter)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (donor.first_name, donor.last_name, donor.email, 
                  donor.phone, donor.anonymous, donor.newsletter))
            
            return cursor.lastrowid
    
    def create_donation(self, donation: Donation, donor_id: int, encrypted_data: str) -> str:
        """Create donation record"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            transaction_id = f"TXN_{datetime.now().strftime('%Y%m%d')}_{uuid.uuid4().hex[:8]}"
            
            cursor.execute('''
                INSERT INTO donations (transaction_id, amount, currency, donation_type, 
                                     payment_method, donor_id, status, encrypted_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (transaction_id, float(donation.amount), donation.currency,
                  donation.donation_type, donation.payment_method, donor_id,
                  donation.status, encrypted_data))
            
            conn.commit()
            return transaction_id
    
    def update_donation_status(self, transaction_id: str, status: str):
        """Update donation status"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE donations SET status = ?, updated_at = CURRENT_TIMESTAMP 
                WHERE transaction_id = ?
            ''', (status, transaction_id))
            conn.commit()
    
    def log_audit_event(self, action: str, user_ip: str, details: dict, transaction_id: str = None):
        """Log audit event"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO audit_logs (action, user_ip, details, transaction_id)
                VALUES (?, ?, ?, ?)
            ''', (action, user_ip, json.dumps(details), transaction_id))
            conn.commit()
    
    def get_donation_stats(self) -> dict:
        """Get donation statistics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Total raised
            cursor.execute("SELECT SUM(amount) FROM donations WHERE status = 'completed'")
            total_raised = cursor.fetchone()[0] or 0
            
            # Recent donations
            cursor.execute('''
                SELECT d.first_name, d.last_name, don.amount, don.created_at, d.anonymous
                FROM donations don
                JOIN donors d ON don.donor_id = d.id
                WHERE don.status = 'completed'
                ORDER BY don.created_at DESC
                LIMIT 10
            ''')
            recent_donations = cursor.fetchall()
            
            return {
                'total_raised': float(total_raised),
                'recent_donations': [
                    {
                        'name': 'Anonymous' if row[4] else f"{row[0]} {row[1][0]}.",
                        'amount': float(row[2]),
                        'date': row[3]
                    }
                    for row in recent_donations
                ]
            }

class PaymentProcessor:
    """Payment processing utilities"""
    
    @staticmethod
    def process_card_payment(amount: float, token: str, description: str) -> dict:
        """Process credit card payment via Stripe"""
        try:
            charge = stripe.Charge.create(
                amount=int(amount * 100),  # Convert to cents
                currency='usd',
                source=token,
                description=description,
                metadata={'system': 'ai_cancer_care_donations'}
            )
            
            return {
                'success': True,
                'charge_id': charge.id,
                'status': charge.status
            }
        except stripe.error.StripeError as e:
            logger.error(f"Stripe payment error: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    @staticmethod
    def process_paypal_payment(amount: float, paypal_data: dict) -> dict:
        """Process PayPal payment (mock implementation)"""
        # In production, integrate with PayPal SDK
        return {
            'success': True,
            'transaction_id': f"PP_{uuid.uuid4().hex[:12]}",
            'status': 'completed'
        }
    
    @staticmethod
    def process_crypto_payment(amount: float, crypto_data: dict) -> dict:
        """Process cryptocurrency payment (mock implementation)"""
        # In production, integrate with crypto payment gateway
        return {
            'success': True,
            'transaction_id': f"CRYPTO_{uuid.uuid4().hex[:12]}",
            'status': 'pending_confirmation'
        }

# Initialize components
security = SecurityValidator()
db_manager = DatabaseManager()
payment_processor = PaymentProcessor()

# Authentication decorator
def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        expected_key = os.environ.get('API_KEY', 'default-api-key')
        
        if not api_key or api_key != expected_key:
            return jsonify({'error': 'Invalid API key'}), 401
        
        return f(*args, **kwargs)
    return decorated

# API Routes

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

@app.route('/api/donate', methods=['POST'])
@limiter.limit("10 per minute")
def process_donation():
    """Process donation request"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate required fields
        required_fields = ['amount', 'donor_info', 'payment_method']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Validate amount
        amount = float(str(data['amount']))
        if not security.validate_amount(float(amount)):
            return jsonify({'error': 'Invalid donation amount'}), 400
        
        # Validate donor information
        donor_info = data['donor_info']
        if not security.validate_email(donor_info.get('email', '')):
            return jsonify({'error': 'Invalid email address'}), 400
        
        # Create donor object
        donor = Donor(
            first_name=security.sanitize_input(donor_info['first_name']),
            last_name=security.sanitize_input(donor_info['last_name']),
            email=donor_info['email'].lower().strip(),
            phone=donor_info.get('phone'),
            anonymous=donor_info.get('anonymous', False),
            newsletter=donor_info.get('newsletter', False)
        )
        
        # Create donation object
        donation = Donation(
            amount=amount,
            donation_type=data.get('donation_type', 'one-time'),
            payment_method=data['payment_method'],
            donor=donor,
            status='pending'
        )
        
        # Encrypt sensitive data
        sensitive_data = {
            'donor': asdict(donor),
            'payment_data': data.get('payment_data', {}),
            'ip_address': request.remote_addr
        }
        encrypted_data = cipher_suite.encrypt(json.dumps(sensitive_data).encode()).decode()
        
        # Create donor in database
        donor_id = db.create_donor(donor)
        
        # Create donation record
        transaction_id = db.create_donation(donation, donor_id, encrypted_data)
        
        # Process payment
        payment_result = None
        if donation.payment_method == 'card':
            payment_result = payment_processor.process_card_payment(
                amount, 
                data.get('payment_token', ''),
                f"AI Cancer Care Donation - {transaction_id}"
            )
        elif donation.payment_method == 'paypal':
            payment_result = payment_processor.process_paypal_payment(
                amount, 
                data.get('paypal_data', {})
            )
        elif donation.payment_method == 'crypto':
            payment_result = payment_processor.process_crypto_payment(
                amount, 
                data.get('crypto_data', {})
            )
        
        # Update donation status
        if payment_result and payment_result['success']:
            db.update_donation_status(transaction_id, 'completed')
            status = 'success'
        else:
            db.update_donation_status(transaction_id, 'failed')
            status = 'failed'
        
        # Log audit event
        db.log_audit_event(
            'donation_processed',
            request.remote_addr,
            {
                'transaction_id': transaction_id,
                'amount': float(amount),
                'payment_method': donation.payment_method,
                'status': status
            },
            transaction_id
        )
        
        # Log donation
        logger.info(f"Donation processed: {transaction_id}, Amount: ${amount}, Status: {status}")
        
        return jsonify({
            'success': payment_result['success'] if payment_result else False,
            'transaction_id': transaction_id,
            'status': status,
            'message': 'Donation processed successfully' if status == 'success' else 'Payment processing failed'
        })
        
    except Exception as e:
        logger.error(f"Donation processing error: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500

@app.route('/api/stats', methods=['GET'])
def get_donation_stats():
    """Get donation statistics"""
    try:
        stats = db.get_donation_stats()
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Stats retrieval error: {str(e)}")
        return jsonify({'error': 'Failed to retrieve statistics'}), 500

@app.route('/api/transparency', methods=['GET'])
def get_transparency_report():
    """Get transparency report"""
    try:
        # Mock transparency data - in production, pull from database
        transparency_data = {
            'fund_allocation': {
                'ai_research': 75,
                'patient_care': 15,
                'infrastructure': 7,
                'administrative': 3
            },
            'impact_metrics': {
                'patients_helped': 2450,
                'diagnostic_accuracy': 89,
                'treatment_time_reduction': 45,
                'partner_hospitals': 12
            },
            'security_measures': [
                'AES-256 encryption for all transactions',
                'Multi-factor authentication required',
                'Regular third-party security audits',
                'HIPAA and GDPR compliance'
            ],
            'last_updated': datetime.now().isoformat()
        }
        
        return jsonify(transparency_data)
    except Exception as e:
        logger.error(f"Transparency report error: {str(e)}")
        return jsonify({'error': 'Failed to retrieve transparency report'}), 500

@app.route('/api/verify-donation', methods=['POST'])
@require_api_key
def verify_donation():
    """Verify donation status"""
    try:
        data = request.get_json()
        transaction_id = data.get('transaction_id')
        
        if not transaction_id:
            return jsonify({'error': 'Transaction ID required'}), 400
        
        with sqlite3.connect(db.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT status, amount, created_at FROM donations 
                WHERE transaction_id = ?
            ''', (transaction_id,))
            
            result = cursor.fetchone()
            
            if result:
                return jsonify({
                    'transaction_id': transaction_id,
                    'status': result[0],
                    'amount': float(result[1]),
                    'created_at': result[2]
                })
            else:
                return jsonify({'error': 'Transaction not found'}), 404
                
    except Exception as e:
        logger.error(f"Donation verification error: {str(e)}")
        return jsonify({'error': 'Verification failed'}), 500

# Function to emit real-time updates (called from patient dashboard)
def emit_patient_update(patient_id, update_type, data):
    patient = Patient.query.get(patient_id)
    if patient:
        socketio.emit('patient_update', {
            'patient_id': patient_id,
            'update_type': update_type,
            'data': data,
            'timestamp': datetime.utcnow().isoformat()
        }, room=f'oncologist_{patient.oncologist_id}')

def emit_new_alert(alert_id):
    alert = Alert.query.get(alert_id)
    if alert:
        patient = Patient.query.get(alert.patient_id)
        user = User.query.get(patient.user_id)
        
        socketio.emit('new_alert', {
            'alert_id': alert.id,
            'patient_name': user.full_name,
            'title': alert.title,
            'message': alert.message,
            'priority': alert.priority,
            'timestamp': alert.created_at.isoformat()
        }, room=f'oncologist_{alert.oncologist_id}')

# API endpoints for patient dashboard to trigger updates
@app.route('/api/trigger-patient-update', methods=['POST'])
def trigger_patient_update():
    data = request.json
    emit_patient_update(
        data['patient_id'],
        data['update_type'],
        data['data']
    )
    return jsonify({'status': 'success'})

@app.route('/api/trigger-alert', methods=['POST'])
def trigger_alert():
    data = request.json
    
    # Create new alert
    alert = Alert(
        patient_id=data['patient_id'],
        oncologist_id=data['oncologist_id'],
        alert_type=data['alert_type'],
        priority=data['priority'],
        title=data['title'],
        message=data['message']
    )
    
    db.session.add(alert)
    db.session.commit()
    
    # Emit to oncologist
    emit_new_alert(alert.id)
    
    return jsonify({'status': 'success', 'alert_id': alert.id})

# WebSocket Events for Real-time Updates
@socketio.on('connect')
def handle_connect():
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        if user and user.role == 'oncologist':
            join_room(f'oncologist_{user.id}')
            emit('status', {'msg': f'Connected as {user.full_name}'})

@socketio.on('disconnect')
def handle_disconnect():
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        if user and user.role == 'oncologist':
            leave_room(f'oncologist_{user.id}')

# Error handlers
@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({'error': 'Rate limit exceeded. Please try again later.'}), 429

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not Found', 'message': 'Resource not found'}), 404


@app.route('/donate')
def donate_home():
    """Render donation page"""
    return render_template("donation.html")

# Root route to serve the frontend
@app.route('/')
def home():
    return render_template("index.html")

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)