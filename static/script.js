// Application State
        let currentUser = null;
        let currentView = 'overview';

         // Form toggle functionality
        const loginBtn = document.getElementById('loginBtn');
        const registerBtn = document.getElementById('registerBtn');
        const loginForm = document.getElementById('loginForm');
        const registerForm = document.getElementById('registerForm');

        loginBtn.addEventListener('click', () => {
            loginForm.classList.remove('hidden');
            registerForm.classList.add('hidden');
            loginBtn.classList.add('bg-white', 'bg-opacity-20');
            loginBtn.classList.remove('text-indigo-200');
            registerBtn.classList.remove('bg-white', 'bg-opacity-20');
            registerBtn.classList.add('text-indigo-200');
        });

        registerBtn.addEventListener('click', () => {
            registerForm.classList.remove('hidden');
            loginForm.classList.add('hidden');
            registerBtn.classList.add('bg-white', 'bg-opacity-20');
            registerBtn.classList.remove('text-indigo-200');
            loginBtn.classList.remove('bg-white', 'bg-opacity-20');
            loginBtn.classList.add('text-indigo-200');
        });

        // Password visibility toggle
        function togglePassword(inputId) {
            const input = document.getElementById(inputId);
            const icon = input.nextElementSibling.querySelector('i');
            
            if (input.type === 'password') {
                input.type = 'text';
                icon.classList.remove('fa-eye');
                icon.classList.add('fa-eye-slash');
            } else {
                input.type = 'password';
                icon.classList.remove('fa-eye-slash');
                icon.classList.add('fa-eye');
            }
        }

        // Password strength checker
        document.getElementById('password').addEventListener('input', function(e) {
            const password = e.target.value;
            const strengthDiv = document.getElementById('passwordStrength');
            const strength = checkPasswordStrength(password);
            
            strengthDiv.innerHTML = `
                <div class="flex items-center space-x-2">
                    <div class="flex space-x-1">
                        ${[1,2,3,4].map(i => `
                            <div class="w-6 h-1 rounded ${i <= strength.score ? strength.color : 'bg-gray-300'}"></div>
                        `).join('')}
                    </div>
                    <span class="text-white text-xs">${strength.text}</span>
                </div>
            `;
        });

        function checkPasswordStrength(password) {
            let score = 0;
            if (password.length >= 8) score++;
            if (/[a-z]/.test(password)) score++;
            if (/[A-Z]/.test(password)) score++;
            if (/[0-9]/.test(password)) score++;
            if (/[^A-Za-z0-9]/.test(password)) score++;

            const levels = [
                { score: 0, text: 'Very Weak', color: 'bg-red-500' },
                { score: 1, text: 'Weak', color: 'bg-red-400' },
                { score: 2, text: 'Fair', color: 'bg-yellow-400' },
                { score: 3, text: 'Good', color: 'bg-blue-400' },
                { score: 4, text: 'Strong', color: 'bg-green-400' },
                { score: 5, text: 'Very Strong', color: 'bg-green-500' }
            ];

            return levels[Math.min(score, 5)];
        }

        // Alert system
        function showAlert(message, type = 'info') {
            const alertContainer = document.getElementById('alertContainer');
            const alertId = 'alert-' + Date.now();
            const alertColors = {
                success: 'bg-green-500',
                error: 'bg-red-500',
                warning: 'bg-yellow-500',
                info: 'bg-blue-500'
            };

            const alertHTML = `
                <div id="${alertId}" class="mb-4 p-4 rounded-lg ${alertColors[type]} text-white shadow-lg transform translate-x-full transition-transform duration-300">
                    <div class="flex items-center justify-between">
                        <span>${message}</span>
                        <button onclick="closeAlert('${alertId}')" class="ml-4 text-white hover:text-gray-200">
                            <i class="fas fa-times"></i>
                        </button>
                    </div>
                </div>
            `;

            alertContainer.insertAdjacentHTML('beforeend', alertHTML);
            
            // Slide in
            setTimeout(() => {
                document.getElementById(alertId).classList.remove('translate-x-full');
            }, 100);

            // Auto remove after 5 seconds
            setTimeout(() => {
                closeAlert(alertId);
            }, 5000);
        }

        function closeAlert(alertId) {
            const alert = document.getElementById(alertId);
            if (alert) {
                alert.classList.add('translate-x-full');
                setTimeout(() => alert.remove(), 300);
            }
        }

        // Form validation
        function validateEmail(email) {
            return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
        }

        function validatePassword(password) {
            return password.length >= 8;
        }

        // Login form submission
        loginForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const email = document.getElementById('email').value;
            const password = document.getElementById('password').value;

            if (!validateEmail(email)) {
                showAlert('Please enter a valid email address.', 'error');
                return;
            }

            if (!validatePassword(password)) {
                showAlert('Password must be at least 8 characters long.', 'error');
                return;
            }

            document.getElementById('loadingOverlay').classList.remove('hidden');

            try {
                const response = await fetch('/api/login', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ email, password })
                });

                const data = await response.json();

                if (response.ok) {
                    showAlert('Login successful! Redirecting...', 'success');
                    localStorage.setItem('authToken', data.token);
                    
                    // Redirect based on user role
                    setTimeout(() => {
                        const dashboardUrls = {
                            patient: '/patient-dashboard',
                            nurse: '/nurse-dashboard',
                            oncologist: '/oncologist-dashboard'
                        };
                        window.location.href = dashboardUrls[data.user.role] || '/dashboard';
                    }, 1500);
                } else {
                    showAlert(data.message || 'Login failed. Please try again.', 'error');
                }
            } catch (error) {
                showAlert('Network error. Please check your connection.', 'error');
            } finally {
                document.getElementById('loadingOverlay').classList.add('hidden');
            }
        });

        // Register form submission
        registerForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const firstName = document.getElementById('firstName').value;
            const lastName = document.getElementById('lastName').value;
            const email = document.getElementById('email').value;
            const role = document.getElementById('role').value;
            const password = document.getElementById('password').value;
            const confirmPassword = document.getElementById('confirmPassword').value;
            const agreeTerms = document.getElementById('agreeTerms').checked;

            // Validation
            if (!firstName || !lastName) {
                showAlert('Please enter your full name.', 'error');
                return;
            }

            if (!validateEmail(email)) {
                showAlert('Please enter a valid email address.', 'error');
                return;
            }

            if (!role) {
                showAlert('Please select your role.', 'error');
                return;
            }

            if (!validatePassword(password)) {
                showAlert('Password must be at least 8 characters long.', 'error');
                return;
            }

            if (password !== confirmPassword) {
                showAlert('Passwords do not match.', 'error');
                return;
            }

            if (!agreeTerms) {
                showAlert('Please agree to the terms and conditions.', 'error');
                return;
            }

            document.getElementById('loadingOverlay').classList.remove('hidden');

            try {
                const response = await fetch('/api/register', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        firstName,
                        lastName,
                        email,
                        role,
                        password
                    })
                });

                const data = await response.json();

                if (response.ok) {
                    showAlert('Registration successful! Please login.', 'success');
                    setTimeout(() => {
                        loginBtn.click();
                    }, 1500);
                } else {
                    showAlert(data.message || 'Registration failed. Please try again.', 'error');
                }
            } catch (error) {
                showAlert('Network error. Please check your connection.', 'error');
            } finally {
                document.getElementById('loadingOverlay').classList.add('hidden');
            }
        });

        // Check for existing authentication on page load
        window.addEventListener('load', function() {
            const token = localStorage.getItem('authToken');
            if (token) {
                // Verify token with backend
                fetch('/api/verify-token', {
                    headers: {
                        'Authorization': `Bearer ${token}`
                    }
                })
                .then(response => response.json())
                .then(data => {
                    if (data.valid) {
                        const dashboardUrls = {
                            patient: '/patient-dashboard',
                            nurse: '/nurse-dashboard',
                            oncologist: '/oncologist-dashboard'
                        };
                        window.location.href = dashboardUrls[data.user.role] || '/dashboard';
                    } else {
                        localStorage.removeItem('authToken');
                    }
                })
                .catch(() => {
                    localStorage.removeItem('authToken');
                });
            }
        });

        // Synthetic Medical Data
        const syntheticData = {
            patients: [
                { id: 1, name: 'Sarah Johnson', age: 45, diagnosis: 'Breast Cancer', stage: 'Stage II', lastVisit: '2024-06-05', riskScore: 0.65 },
                { id: 2, name: 'Michael Chen', age: 62, diagnosis: 'Lung Cancer', stage: 'Stage IIIA', lastVisit: '2024-06-08', riskScore: 0.78 },
                { id: 3, name: 'Emily Rodriguez', age: 38, diagnosis: 'Cervical Cancer', stage: 'Stage I', lastVisit: '2024-06-10', riskScore: 0.42 },
                { id: 4, name: 'David Thompson', age: 55, diagnosis: 'Colon Cancer', stage: 'Stage III', lastVisit: '2024-06-09', riskScore: 0.71 }
            ],
            
            treatments: [
                { patientId: 1, treatment: 'Chemotherapy', cycle: 4, totalCycles: 8, nextDate: '2024-06-15' },
                { patientId: 2, treatment: 'Radiation', session: 12, totalSessions: 25, nextDate: '2024-06-12' },
                { patientId: 3, treatment: 'Surgery', status: 'Completed', followUp: '2024-06-20' },
                { patientId: 4, treatment: 'Immunotherapy', cycle: 2, totalCycles: 6, nextDate: '2024-06-18' }
            ],
            
            medications: [
                { name: 'Doxorubicin', dosage: '60mg/m²', frequency: 'Every 3 weeks', status: 'taken' },
                { name: 'Cyclophosphamide', dosage: '600mg/m²', frequency: 'Every 3 weeks', status: 'pending' },
                { name: 'Tamoxifen', dosage: '20mg', frequency: 'Daily', status: 'taken' },
                { name: 'Ondansetron', dosage: '8mg', frequency: 'As needed', status: 'missed' }
            ],
            
            appointments: [
                { date: '2024-06-12', time: '10:00 AM', type: 'Oncology Consultation', doctor: 'Dr. Smith' },
                { date: '2024-06-15', time: '2:00 PM', type: 'Chemotherapy', doctor: 'Nurse Johnson' },
                { date: '2024-06-20', time: '11:30 AM', type: 'Blood Work', doctor: 'Lab Tech' },
                { date: '2024-06-25', time: '9:00 AM', type: 'Follow-up', doctor: 'Dr. Smith' }
            ]
        };

        // AI/ML Simulation Functions
        function calculateRiskScore(patient) {
            // Simulated ML model for risk assessment
            const ageWeight = patient.age > 60 ? 0.3 : 0.1;
            const stageWeight = patient.stage.includes('III') ? 0.4 : patient.stage.includes('II') ? 0.3 : 0.2;
            const typeWeight = patient.diagnosis.includes('Lung') ? 0.4 : 0.2;
            
            return Math.min(ageWeight + stageWeight + typeWeight + Math.random() * 0.2, 1.0);
        }

        function generateAIInsights(role) {
            const insights = {
                patient: [
                    "Your recent lab results show improvement in white blood cell count.",
                    "Based on your symptoms, consider scheduling a nutrition consultation.",
                    "Your treatment adherence rate is excellent at 95%."
                ],
                nurse: [
                    "3 patients require medication schedule adjustments this week.",
                    "Patient compliance rates have improved by 12% this month.",
                    "Alert: Sarah Johnson missed her last appointment - follow-up needed."
                ],
                oncologist: [
                    "Treatment response prediction suggests 85% success rate for current protocol.",
                    "New clinical trial may benefit 2 of your current patients.",
                    "Risk stratification model identifies 1 high-risk patient requiring immediate attention."
                ],
                admin: [
                    "Platform usage increased 23% this month across all user types.",
                    "Patient satisfaction scores average 4.7/5.0 system-wide.",
                    "Resource utilization optimized - 15% reduction in wait times."
                ]
            };
            
            return insights[role] || [];
        }

        // Authentication System
        function login(event) {
            event.preventDefault();
            
            const role = document.getElementById('role').value;
            const username = document.getElementById('firstName').value;
            const password = document.getElementById('password').value;
            
            // Show loading state
            document.getElementById('loginText').classList.add('hidden');
            document.getElementById('loginLoading').classList.remove('hidden');
            
            // Simulate authentication delay
            setTimeout(() => {
                // Simple demo authentication
                const validCredentials = {
                    'patient123': { role: 'patient', name: 'Sarah Johnson' },
                    'nurse123': { role: 'nurse', name: 'Jennifer Martinez' },
                    'doctor123': { role: 'oncologist', name: 'Dr. Michael Smith' },
                    'admin123': { role: 'admin', name: 'System Administrator' }
                };
                
                if (validCredentials[username] && password === 'demo123' && validCredentials[username].role === role) {
                    currentUser = {
                        username,
                        role,
                        name: validCredentials[username].name
                    };
                    
                    showDashboard();
                } else {
                    alert('Invalid credentials. Please check the demo credentials provided.');
                }
                
                // Hide loading state
                document.getElementById('loginText').classList.remove('hidden');
                document.getElementById('loginLoading').classList.add('hidden');
            }, 1500);
        }

        function logout() {
            currentUser = null;
            document.getElementById('dashboard').style.display = 'none';
            document.getElementById('loginScreen').style.display = 'flex';
        }

        // Dashboard Functions
        function showDashboard() {
            document.getElementById('loginScreen').style.display = 'none';
            document.getElementById('dashboard').style.display = 'grid';
            
            setupSidebar();
            showOverview();
        }

        function setupSidebar() {
            const menus = {
                patient: [
                    { id: 'overview', text: '📊 Dashboard', icon: '📊' },
                    { id: 'treatments', text: '💊 My Treatments', icon: '💊' },
                    { id: 'appointments', text: '📅 Appointments', icon: '📅' },
                    { id: 'medications', text: '💉 Medications', icon: '💉' },
                    { id: 'reports', text: '📋 Reports', icon: '📋' }
                ],
                nurse: [
                    { id: 'overview', text: '📊 Dashboard', icon: '📊' },
                    { id: 'patients', text: '👥 My Patients', icon: '👥' },
                    { id: 'schedules', text: '📅 Schedules', icon: '📅' },
                    { id: 'medications', text: '💊 Medication Tracking', icon: '💊' },
                    { id: 'communications', text: '💬 Messages', icon: '💬' }
                ],
                oncologist: [
                    { id: 'overview', text: '📊 Dashboard', icon: '📊' },
                    { id: 'patients', text: '👥 Patient Overview', icon: '👥' },
                    { id: 'treatments', text: '🔬 Treatment Plans', icon: '🔬' },
                    { id: 'research', text: '📊 Research Data', icon: '📊' },
                    { id: 'ai-insights', text: '🤖 AI Insights', icon: '🤖' }
                ],
                admin: [
                    { id: 'overview', text: '📊 Dashboard', icon: '📊' },
                    { id: 'users', text: '👥 User Management', icon: '👥' },
                    { id: 'analytics', text: '📈 Analytics', icon: '📈' },
                    { id: 'system', text: '⚙️ System Config', icon: '⚙️' },
                    { id: 'reports', text: '📋 Reports', icon: '📋' }
                ]
            };

            const sidebarMenu = document.getElementById('sidebarMenu');
            sidebarMenu.innerHTML = '';
            
            menus[currentUser.role].forEach(item => {
                const li = document.createElement('li');
                li.innerHTML = `<a href="#" onclick="navigateTo('${item.id}')" class="${item.id === currentView ? 'active' : ''}">${item.text}</a>`;
                sidebarMenu.appendChild(li);
            });
        }

        function navigateTo(view) {
            currentView = view;
            setupSidebar(); // Refresh active state
            
            switch(view) {
                case 'overview':
                    showOverview();
                    break;
                case 'patients':
                    showPatients();
                    break;
                case 'treatments':
                    showTreatments();
                    break;
                case 'appointments':
                    showAppointments();
                    break;
                case 'medications':
                    showMedications();
                    break;
                case 'ai-insights':
                    showAIInsights();
                    break;
                default:
                    showOverview();
            }
        }

        function showOverview() {
            const insights = generateAIInsights(currentUser.role);
            const content = document.getElementById('dashboardContent');
            
            content.innerHTML = `
                <h1 style="margin-bottom: 2rem;">Welcome back, ${currentUser.name}</h1>
                
                <div class="dashboard-grid">
                    ${generateOverviewCards()}
                </div>
                
                <div class="ai-insight">
                    <div class="ai-insight-header">
                        <span class="ai-badge">AI</span>
                        <strong>Personalized Insights</strong>
                    </div>
                    ${insights.map(insight => `<p style="margin-bottom: 0.5rem;">• ${insight}</p>`).join('')}
                </div>
                
                ${generateRoleSpecificContent()}
            `;
        }

        function generateOverviewCards() {
            const cards = {
                patient: `
                    <div class="card">
                        <div class="card-header">
                            <span class="card-title">Treatment Progress</span>
                            <div class="card-icon icon-blue">💊</div>
                        </div>
                        <div style="font-size: 2rem; font-weight: bold; color: var(--primary-blue);">65%</div>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: 65%;"></div>
                        </div>
                        <p style="margin-top: 0.5rem; color: var(--text-light);">4 of 8 cycles completed</p>
                    </div>
                    
                    <div class="card">
                        <div class="card-header">
                            <span class="card-title">Next Appointment</span>
                            <div class="card-icon icon-green">📅</div>
                        </div>
                        <div style="font-size: 1.2rem; font-weight: bold; margin-bottom: 0.5rem;">June 15, 2024</div>
                        <p style="color: var(--text-light);">Chemotherapy Session</p>
                        <p style="color: var(--text-light);">Dr. Smith - 2:00 PM</p>
                    </div>
                    
                    <div class="card">
                        <div class="card-header">
                            <span class="card-title">Health Score</span>
                            <div class="card-icon icon-orange">❤️</div>
                        </div>
                        <div style="font-size: 2rem; font-weight: bold; color: var(--primary-green);">Good</div>
                        <p style="color: var(--text-light);">Based on recent vitals and lab results</p>
                    </div>
                `,
                
                nurse: `
                    <div class="card">
                        <div class="card-header">
                            <span class="card-title">Active Patients</span>
                            <div class="card-icon icon-blue">👥</div>
                        </div>
                        <div style="font-size: 2rem; font-weight: bold; color: var(--primary-blue);">12</div>
                        <p style="color: var(--text-light);">3 requiring immediate attention</p>
                    </div>
                    
                    <div class="card">
                        <div class="card-header">
                            <span class="card-title">Today's Schedule</span>
                            <div class="card-icon icon-green">📅</div>
                        </div>
                        <div style="font-size: 2rem; font-weight: bold; color: var(--primary-green);">8</div>
                        <p style="color: var(--text-light);">appointments scheduled</p>
                    </div>
                    
                    <div class="card">
                        <div class="card-header">
                            <span class="card-title">Medication Alerts</span>
                            <div class="card-icon icon-red">⚠️</div>
                        </div>
                        <div style="font-size: 2rem; font-weight: bold; color: var(--red-500);">3</div>
                        <p style="color: var(--text-light);">patients need follow-up</p>
                    </div>
                `,
                
                oncologist: `
                    <div class="card">
                        <div class="card-header">
                            <span class="card-title">Patient Caseload</span>
                            <div class="card-icon icon-blue">👥</div>
                        </div>
                        <div style="font-size: 2rem; font-weight: bold; color: var(--primary-blue);">24</div>
                        <p style="color: var(--text-light);">active cancer patients</p>
                    </div>
                    
                    <div class="card">
                        <div class="card-header">
                            <span class="card-title">Treatment Success Rate</span>
                            <div class="card-icon icon-green">📊</div>
                        </div>
                        <div style="font-size: 2rem; font-weight: bold; color: var(--primary-green);">87%</div>
                        <p style="color: var(--text-light);">based on AI predictions</p>
                    </div>
                    
                    <div class="card">
                        <div class="card-header">
                            <span class="card-title">Research Insights</span>
                            <div class="card-icon icon-orange">🔬</div>
                        </div>
                        <div style="font-size: 2rem; font-weight: bold; color: var(--orange-500);">5</div>
                        <p style="color: var(--text-light);">new clinical trials available</p>
                    </div>
                `,
                
                admin: `
                    <div class="card">
                        <div class="card-header">
                            <span class="card-title">Total Users</span>
                            <div class="card-icon icon-blue">👥</div>
                        </div>
                        <div style="font-size: 2rem; font-weight: bold; color: var(--primary-blue);">1,247</div>
                        <p style="color: var(--text-light);">across all user types</p>
                    </div>
                    
                    <div class="card">
                        <div class="card-header">
                            <span class="card-title">System Health</span>
                            <div class="card-icon icon-green">💚</div>
                        </div>
                        <div style="font-size: 2rem; font-weight: bold; color: var(--primary-green);">99.8%</div>
                        <p style="color: var(--text-light);">uptime this month</p>
                    </div>
                    
                    <div class="card">
                        <div class="card-header">
                            <span class="card-title">Data Processed</span>
                            <div class="card-icon icon-orange">📊</div>
                        </div>
                        <div style="font-size: 2rem; font-weight: bold; color: var(--orange-500);">2.4TB</div>
                        <p style="color: var(--text-light);">medical data analyzed</p>
                    </div>
                `
            };
            
            return cards[currentUser.role] || '';
        }

        function generateRoleSpecificContent() {
            switch(currentUser.role) {
                case 'patient':
                    return `
                        <div class="dashboard-grid">
                            <div class="card">
                                <div class="card-header">
                                    <span class="card-title">Upcoming Appointments</span>
                                    <div class="card-icon icon-blue">📅</div>
                                </div>
                                ${syntheticData.appointments.slice(0, 3).map(apt => `
                                    <div class="appointment-item">
                                        <div>
                                            <div class="appointment-time">${apt.date} - ${apt.time}</div>
                                            <div class="appointment-type">${apt.type}</div>
                                        </div>
                                    </div>
                                `).join('')}
                            </div>
                            
                            <div class="card">
                                <div class="card-header">
                                    <span class="card-title">Current Medications</span>
                                    <div class="card-icon icon-green">💊</div>
                                </div>
                                ${syntheticData.medications.slice(0, 3).map(med => `
                                    <div class="medication-item">
                                        <div>
                                            <strong>${med.name}</strong><br>
                                            <small>${med.dosage} - ${med.frequency}</small>
                                        </div>
                                        <div class="medication-status status-${med.status}"></div>
                                    </div>
                                `).join('')}
                            </div>
                        </div>
                    `;
                    
                case 'nurse':
                case 'oncologist':
                    return `
                        <div class="card">
                            <div class="card-header">
                                <span class="card-title">Recent Patients</span>
                                <div class="card-icon icon-blue">👥</div>
                            </div>
                            <div class="patient-list">
                                ${syntheticData.patients.map(patient => `
                                    <div class="patient-item">
                                        <div class="patient-avatar">${patient.name.charAt(0)}</div>
                                        <div class="patient-info">
                                            <h4>${patient.name}</h4>
                                            <p>${patient.diagnosis} - ${patient.stage}</p>
                                            <p>Last visit: ${patient.lastVisit}</p>
                                        </div>
                                        <div style="margin-left: auto;">
                                            <span class="text-${patient.riskScore > 0.7 ? 'danger' : patient.riskScore > 0.5 ? 'warning' : 'success'}">
                                                Risk: ${Math.round(patient.riskScore * 100)}%
                                            </span>
                                        </div>
                                    </div>
                                `).join('')}
                            </div>
                        </div>
                    `;
                    
                case 'admin':
                    return `
                        <div class="card">
                            <div class="card-header">
                                <span class="card-title">System Analytics</span>
                                <div class="card-icon icon-blue">📈</div>
                            </div>
                            <div class="chart-container">
                                <div style="text-align: center; color: var(--text-light);">
                                    📊 Real-time Analytics Dashboard<br>
                                    <small>User activity, system performance, and health metrics</small>
                                </div>
                            </div>
                        </div>
                    `;
                    
                default:
                    return '';
            }
        }

        function showPatients() {
            const content = document.getElementById('dashboardContent');
            content.innerHTML = `
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 2rem;">
                    <h1>Patient Management</h1>
                    <button class="btn btn-primary" onclick="showAddPatientModal()">+ Add Patient</button>
                </div>
                
                <div class="card">
                    <div class="card-header">
                        <span class="card-title">Active Patients</span>
                        <div style="display: flex; gap: 1rem;">
                            <input type="text" placeholder="Search patients..." style="padding: 0.5rem; border-radius: 6px; border: 1px solid var(--gray-300);">
                            <select style="padding: 0.5rem; border-radius: 6px; border: 1px solid var(--gray-300);">
                                <option>All Stages</option>
                                <option>Stage I</option>
                                <option>Stage II</option>
                                <option>Stage III</option>
                                <option>Stage IV</option>
                            </select>
                        </div>
                    </div>
                    
                    <div style="overflow-x: auto;">
                        <table style="width: 100%; border-collapse: collapse; margin-top: 1rem;">
                            <thead>
                                <tr style="background-color: var(--gray-100);">
                                    <th style="padding: 1rem; text-align: left; border-bottom: 1px solid var(--gray-200);">Patient</th>
                                    <th style="padding: 1rem; text-align: left; border-bottom: 1px solid var(--gray-200);">Diagnosis</th>
                                    <th style="padding: 1rem; text-align: left; border-bottom: 1px solid var(--gray-200);">Stage</th>
                                    <th style="padding: 1rem; text-align: left; border-bottom: 1px solid var(--gray-200);">Risk Score</th>
                                    <th style="padding: 1rem; text-align: left; border-bottom: 1px solid var(--gray-200);">Last Visit</th>
                                    <th style="padding: 1rem; text-align: left; border-bottom: 1px solid var(--gray-200);">Actions</th>
                                </tr>
                            </thead>
                            <tbody>
                                ${syntheticData.patients.map(patient => `
                                    <tr style="border-bottom: 1px solid var(--gray-200);">
                                        <td style="padding: 1rem;">
                                            <div style="display: flex; align-items: center;">
                                                <div class="patient-avatar" style="margin-right: 0.5rem;">${patient.name.charAt(0)}</div>
                                                <div>
                                                    <strong>${patient.name}</strong><br>
                                                    <small>Age: ${patient.age}</small>
                                                </div>
                                            </div>
                                        </td>
                                        <td style="padding: 1rem;">${patient.diagnosis}</td>
                                        <td style="padding: 1rem;">${patient.stage}</td>
                                        <td style="padding: 1rem;">
                                            <span class="text-${patient.riskScore > 0.7 ? 'danger' : patient.riskScore > 0.5 ? 'warning' : 'success'}">
                                                ${Math.round(patient.riskScore * 100)}%
                                            </span>
                                        </td>
                                        <td style="padding: 1rem;">${patient.lastVisit}</td>
                                        <td style="padding: 1rem;">
                                            <button class="btn btn-primary" style="font-size: 0.8rem; padding: 0.3rem 0.6rem;" onclick="viewPatientDetails(${patient.id})">View</button>
                                        </td>
                                    </tr>
                                `).join('')}
                            </tbody>
                        </table>
                    </div>
                </div>
            `;
        }

        function showTreatments() {
            const content = document.getElementById('dashboardContent');
            content.innerHTML = `
                <h1 style="margin-bottom: 2rem;">Treatment Management</h1>
                
                <div class="dashboard-grid">
                    ${syntheticData.treatments.map(treatment => {
                        const patient = syntheticData.patients.find(p => p.id === treatment.patientId);
                        const progress = treatment.cycle && treatment.totalCycles ? 
                            Math.round((treatment.cycle / treatment.totalCycles) * 100) : 100;
                        
                        return `
                            <div class="card">
                                <div class="card-header">
                                    <span class="card-title">${patient.name}</span>
                                    <div class="card-icon icon-blue">💊</div>
                                </div>
                                <div style="margin-bottom: 1rem;">
                                    <strong>${treatment.treatment}</strong><br>
                                    <small>${patient.diagnosis} - ${patient.stage}</small>
                                </div>
                                
                                ${treatment.cycle ? `
                                    <div style="margin-bottom: 1rem;">
                                        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                                            <span>Progress</span>
                                            <span>${progress}%</span>
                                        </div>
                                        <div class="progress-bar">
                                            <div class="progress-fill" style="width: ${progress}%;"></div>
                                        </div>
                                        <small style="color: var(--text-light);">
                                            ${treatment.cycle} of ${treatment.totalCycles} ${treatment.treatment.includes('Chemotherapy') ? 'cycles' : 'sessions'}
                                        </small>
                                    </div>
                                ` : `
                                    <div style="margin-bottom: 1rem;">
                                        <span class="text-success">✓ ${treatment.status}</span>
                                    </div>
                                `}
                                
                                <div style="margin-top: 1rem;">
                                    <strong>Next:</strong> ${treatment.nextDate || treatment.followUp}<br>
                                    <button class="btn btn-primary" style="margin-top: 0.5rem; font-size: 0.8rem; padding: 0.3rem 0.6rem;">
                                        Update Plan
                                    </button>
                                </div>
                            </div>
                        `;
                    }).join('')}
                </div>
                
                <div class="ai-insight">
                    <div class="ai-insight-header">
                        <span class="ai-badge">AI</span>
                        <strong>Treatment Optimization Recommendations</strong>
                    </div>
                    <p>• Patient #2 (Michael Chen) shows 15% better response than predicted - consider treatment intensification</p>
                    <p>• Patient #1 (Sarah Johnson) may benefit from nutritional support during next cycle</p>
                    <p>• New immunotherapy protocol available for Stage III patients - 2 candidates identified</p>
                </div>
            `;
        }

        function showAppointments() {
            const content = document.getElementById('dashboardContent');
            content.innerHTML = `
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 2rem;">
                    <h1>Appointment Management</h1>
                    <button class="btn btn-primary" onclick="scheduleAppointment()">+ Schedule Appointment</button>
                </div>
                
                <div class="dashboard-grid">
                    <div class="card">
                        <div class="card-header">
                            <span class="card-title">Today's Schedule</span>
                            <div class="card-icon icon-blue">📅</div>
                        </div>
                        ${syntheticData.appointments.filter(apt => apt.date === '2024-06-12').map(apt => `
                            <div class="appointment-item">
                                <div>
                                    <div class="appointment-time">${apt.time}</div>
                                    <div class="appointment-type">${apt.type}</div>
                                    <small style="color: var(--text-light);">${apt.doctor}</small>
                                </div>
                                <button class="btn btn-primary" style="font-size: 0.7rem; padding: 0.25rem 0.5rem;">Details</button>
                            </div>
                        `).join('')}
                        
                        <div style="margin-top: 1rem; text-align: center;">
                            <button class="btn btn-secondary" style="width: 100%;">View Full Calendar</button>
                        </div>
                    </div>
                    
                    <div class="card">
                        <div class="card-header">
                            <span class="card-title">Upcoming Appointments</span>
                            <div class="card-icon icon-green">⏰</div>
                        </div>
                        ${syntheticData.appointments.filter(apt => apt.date !== '2024-06-12').slice(0, 4).map(apt => `
                            <div class="appointment-item">
                                <div>
                                    <div class="appointment-time">${apt.date} - ${apt.time}</div>
                                    <div class="appointment-type">${apt.type}</div>
                                    <small style="color: var(--text-light);">${apt.doctor}</small>
                                </div>
                            </div>
                        `).join('')}
                    </div>
                </div>
                
                <div class="ai-insight">
                    <div class="ai-insight-header">
                        <span class="ai-badge">AI</span>
                        <strong>Smart Scheduling Recommendations</strong>
                    </div>
                    <p>• Optimal appointment time for chemotherapy: Tuesday mornings (based on patient energy patterns)</p>
                    <p>• 3 appointment slots can be optimized to reduce patient wait times by 12 minutes average</p>
                    <p>• Consider grouping lab work with oncology consultations for improved efficiency</p>
                </div>
            `;
        }

        function showMedications() {
            const content = document.getElementById('dashboardContent');
            content.innerHTML = `
                <h1 style="margin-bottom: 2rem;">Medication Management</h1>
                
                <div class="dashboard-grid">
                    <div class="card">
                        <div class="card-header">
                            <span class="card-title">Current Medications</span>
                            <div class="card-icon icon-blue">💊</div>
                        </div>
                        ${syntheticData.medications.map(med => `
                            <div class="medication-item">
                                <div>
                                    <strong>${med.name}</strong><br>
                                    <small>${med.dosage} - ${med.frequency}</small>
                                </div>
                                <div style="display: flex; align-items: center; gap: 0.5rem;">
                                    <span class="text-${med.status === 'taken' ? 'success' : med.status === 'pending' ? 'warning' : 'danger'}">
                                        ${med.status === 'taken' ? '✓ Taken' : med.status === 'pending' ? '⏰ Pending' : '⚠️ Missed'}
                                    </span>
                                    <div class="medication-status status-${med.status}"></div>
                                </div>
                            </div>
                        `).join('')}
                        
                        <div style="margin-top: 1rem;">
                            <button class="btn btn-primary" style="width: 100%;">Update Medication Status</button>
                        </div>
                    </div>
                    
                    <div class="card">
                        <div class="card-header">
                            <span class="card-title">Adherence Tracking</span>
                            <div class="card-icon icon-green">📊</div>
                        </div>
                        <div style="text-align: center; margin: 2rem 0;">
                            <div style="font-size: 3rem; font-weight: bold; color: var(--primary-green);">92%</div>
                            <p style="color: var(--text-light);">Overall Adherence Rate</p>
                        </div>
                        
                        <div style="margin-top: 1rem;">
                            <h4 style="margin-bottom: 0.5rem;">Weekly Progress</h4>
                            <div class="chart-container" style="height: 100px;">
                                <div style="text-align: center; color: var(--text-light);">
                                    📈 Adherence trend: +5% this week
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="card">
                        <div class="card-header">
                            <span class="card-title">Side Effects Monitor</span>
                            <div class="card-icon icon-orange">⚠️</div>
                        </div>
                        <div style="margin: 1rem 0;">
                            <div style="margin-bottom: 1rem;">
                                <strong>Recent Reports:</strong>
                                <ul style="margin-top: 0.5rem; padding-left: 1.5rem;">
                                    <li>Mild nausea (Day 3 post-chemo)</li>
                                    <li>Fatigue (Manageable level)</li>
                                    <li>Slight appetite decrease</li>
                                </ul>
                            </div>
                            
                            <button class="btn btn-secondary" style="width: 100%;">Report Side Effects</button>
                        </div>
                    </div>
                </div>
                
                <div class="ai-insight">
                    <div class="ai-insight-header">
                        <span class="ai-badge">AI</span>
                        <strong>Medication Insights</strong>
                    </div>
                    <p>• Consider taking Ondansetron 30 minutes before meals to improve effectiveness</p>
                    <p>• Doxorubicin levels are optimal - continue current dosing schedule</p>
                    <p>• Reminder set for Tamoxifen - best taken at the same time daily for consistency</p>
                </div>
            `;
        }

        function showAIInsights() {
            const content = document.getElementById('dashboardContent');
            content.innerHTML = `
                <h1 style="margin-bottom: 2rem;">AI-Powered Healthcare Insights</h1>
                
                <div class="ai-insight">
                    <div class="ai-insight-header">
                        <span class="ai-badge">PREDICTIVE</span>
                        <strong>Treatment Response Prediction</strong>
                    </div>
                    <p>Based on patient genomics, treatment history, and real-time biomarkers, our AI model predicts:</p>
                    <ul style="margin-left: 1.5rem; margin-top: 0.5rem;">
                        <li>Patient Sarah Johnson: 85% probability of positive response to current protocol</li>
                        <li>Patient Michael Chen: Consider combination therapy - 23% improvement predicted</li>
                        <li>Patient Emily Rodriguez: Excellent prognosis - 95% 5-year survival probability</li>
                    </ul>
                </div>
                
                <div class="dashboard-grid">
                    <div class="card">
                        <div class="card-header">
                            <span class="card-title">Risk Stratification</span>
                            <div class="card-icon icon-red">⚠️</div>
                        </div>
                        <div style="margin: 1rem 0;">
                            <h4 style="color: var(--red-500); margin-bottom: 0.5rem;">High Risk Patients</h4>
                            <div style="background: var(--light-blue); padding: 0.5rem; border-radius: 6px; margin-bottom: 0.5rem;">
                                <strong>Michael Chen</strong> - Risk Score: 78%<br>
                                <small>Recommend immediate intervention review</small>
                            </div>
                            
                            <h4 style="color: var(--orange-500); margin-bottom: 0.5rem;">Medium Risk</h4>
                            <div style="background: var(--light-green); padding: 0.5rem; border-radius: 6px;">
                                <strong>David Thompson</strong> - Risk Score: 71%<br>
                                <small>Monitor closely, adjust treatment as needed</small>
                            </div>
                        </div>
                    </div>
                    
                    <div class="card">
                        <div class="card-header">
                            <span class="card-title">Biomarker Analysis</span>
                            <div class="card-icon icon-blue">🧬</div>
                        </div>
                        <div style="margin: 1rem 0;">
                            <h4 style="margin-bottom: 0.5rem;">Key Findings</h4>
                            <ul style="font-size: 0.9rem;">
                                <li><strong>CA 15-3:</strong> Trending downward (Good)</li>
                                <li><strong>CEA levels:</strong> Within normal range</li>
                                <li><strong>Circulating tumor DNA:</strong> Decreased 40%</li>
                                <li><strong>Immune markers:</strong> Showing positive response</li>
                            </ul>
                            
                            <div class="progress-bar" style="margin-top: 1rem;">
                                <div class="progress-fill" style="width: 76%;"></div>
                            </div>
                            <small style="color: var(--text-light);">Overall biomarker improvement: 76%</small>
                        </div>
                    </div>
                    
                    <div class="card">
                        <div class="card-header">
                            <span class="card-title">Clinical Trial Matching</span>
                            <div class="card-icon icon-green">🔬</div>
                        </div>
                        <div style="margin: 1rem 0;">
                            <h4 style="margin-bottom: 0.5rem;">Available Trials</h4>
                            <div style="background: var(--light-blue); padding: 0.75rem; border-radius: 6px; margin-bottom: 0.5rem;">
                                <strong>NCT04567890</strong><br>
                                <small>Immunotherapy + Targeted Therapy</small><br>
                                <small style="color: var(--primary-blue);">Match: Michael Chen (Stage IIIA Lung Cancer)</small>
                            </div>
                            
                            <div style="background: var(--light-green); padding: 0.75rem; border-radius: 6px;">
                                <strong>NCT04789123</strong><br>
                                <small>Precision Medicine Study</small><br>
                                <small style="color: var(--primary-green);">Match: Sarah Johnson (HER2+ Breast Cancer)</small>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="card">
                    <div class="card-header">
                        <span class="card-title">Machine Learning Model Performance</span>
                        <div class="card-icon icon-blue">🤖</div>
                    </div>
                    
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-top: 1rem;">
                        <div style="text-align: center; padding: 1rem; background: var(--light-blue); border-radius: 8px;">
                            <div style="font-size: 1.5rem; font-weight: bold; color: var(--primary-blue);">94.2%</div>
                            <small>Prediction Accuracy</small>
                        </div>
                        
                        <div style="text-align: center; padding: 1rem; background: var(--light-green); border-radius: 8px;">
                            <div style="font-size: 1.5rem; font-weight: bold; color: var(--primary-green);">87.6%</div>
                            <small>Early Detection Rate</small>
                        </div>
                        
                        <div style="text-align: center; padding: 1rem; background: var(--gray-100); border-radius: 8px;">
                            <div style="font-size: 1.5rem; font-weight: bold; color: var(--orange-500);">15.3%</div>
                            <small>Treatment Time Reduction</small>
                        </div>
                        
                        <div style="text-align: center; padding: 1rem; background: var(--light-blue); border-radius: 8px;">
                            <div style="font-size: 1.5rem; font-weight: bold; color: var(--primary-blue);">2,847</div>
                            <small>Patients Analyzed</small>
                        </div>
                    </div>
                </div>
            `;
        }

        // Utility Functions
        function viewPatientDetails(patientId) {
            const patient = syntheticData.patients.find(p => p.id === patientId);
            alert(`Viewing details for ${patient.name}\n\nDiagnosis: ${patient.diagnosis}\nStage: ${patient.stage}\nAge: ${patient.age}\nRisk Score: ${Math.round(patient.riskScore * 100)}%\n\nThis would open a detailed patient record in a full implementation.`);
        }

        function showProfile() {
            alert(`Profile Information:\n\nName: ${currentUser.name}\nRole: ${currentUser.role.charAt(0).toUpperCase() + currentUser.role.slice(1)}\nUsername: ${currentUser.username}\n\nThis would open a detailed profile management interface in a full implementation.`);
        }

        function scheduleAppointment() {
            alert('Schedule Appointment\n\nThis would open a comprehensive appointment scheduling interface with:\n• Calendar integration\n• Doctor availability\n• Patient preferences\n• Automated reminders\n• AI-optimized time slots');
        }

        function showAddPatientModal() {
            alert('Add New Patient\n\nThis would open a detailed patient registration form with:\n• Personal information\n• Medical history\n• Insurance details\n• Emergency contacts\n• Initial assessment data');
        }

        // Initialize the application
        document.addEventListener('DOMContentLoaded', function() {
            // Show login screen by default
            document.getElementById('loginScreen').style.display = 'flex';
            document.getElementById('dashboard').style.display = 'none';
        });

        // Simulate real-time updates
        setInterval(() => {
            if (currentUser && currentView === 'overview') {
                // Simulate real-time data updates
                const elements = document.querySelectorAll('.progress-fill');
                elements.forEach(el => {
                    const currentWidth = parseInt(el.style.width) || 0;
                    const variation = (Math.random() - 0.5) * 2; // Small random variation
                    const newWidth = Math.max(0, Math.min(100, currentWidth + variation));
                    el.style.width = newWidth + '%';
                });
            }
        }, 10000); // Update every 10 seconds

        // Handle responsive design
        window.addEventListener('resize', function() {
            if (window.innerWidth <= 768) {
                document.querySelector('.sidebar').style.display = 'none';
            } else if (currentUser) {
                document.querySelector('.sidebar').style.display = 'block';
            }
        });