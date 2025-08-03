Dynamic Pricing using PPO: A 
Reinforcement Learning Project 
This report details a reinforcement learning project focused on developing a dynamic pricing 
agent using the Proximal Policy Optimization (PPO) algorithm. The project encompasses 
environment design, model training, API development, and an interactive dashboard for 
practical application. 
1. 
�
�
 Introduction 
Dynamic pricing is a strategy where businesses adjust prices for products or services in 
real-time based on market demand, supply, competitor pricing, and other external factors. Its 
importance lies in its ability to optimize revenue, maximize profits, and efficiently manage 
inventory in fluctuating market conditions. 
Real-world applications of dynamic pricing are widespread, including: 
●  Airlines and Hotels:  Prices for seats and rooms constantly change based on booking 
time, demand, and availability. 
●  E-commerce Platforms:  Online retailers frequently adjust product prices based on 
Browse behavior, competitor prices, and stock levels. 
●  Ride-sharing Services:  Surge pricing during peak hours or high demand. 
This project aims to achieve: 
●  Develop a simulated environment  for dynamic pricing. 
●  Train a reinforcement learning agent  using the PPO algorithm to learn optimal pricing 
strategies. 
●  Build a robust API  for price recommendations. 
●  Create an interactive dashboard  to visualize and interact with the pricing model in 
real-time. 
2. 
�
�
 Core Concepts 
What is Reinforcement Learning? 
Reinforcement Learning (RL) is a paradigm of machine learning where an agent learns to 
make decisions by performing actions in an environment to maximize a cumulative reward. 
The agent receives feedback in the form of rewards or penalties, which guides it to discover 
the best sequence of actions. 
Overview of PPO (Proximal Policy Optimization) Algorithm 
PPO is a popular on-policy, actor-critic reinforcement learning algorithm that strikes a 
balance between ease of implementation, sample efficiency, and performance. It works by 
optimizing a "clipped" surrogate objective function, which prevents the new policy from 
straying too far from the old policy, ensuring stable and efficient learning. PPO is well-suited 
for environments with continuous action spaces, like price setting in this project. 
What is an Environment in RL, and how it applies to pricing? 
In RL, the environment is the simulated world where the agent operates and learns. For this 
project, the PricingEnvironment simulates a business scenario where a product is sold over 
time. 
●  States (Observations):  The agent observes the current state of the environment. In 
PricingEnvironment, the observation space is a 2-dimensional array representing: 
○  Normalized Inventory:  Current inventory level divided by the maximum inventory. 
○  Normalized Demand Factor:  A scaled value indicating the general demand, 
normalized between 0 and 1 (original range 0.7 to 1.3). 
○  The observation space for the model loaded by pricing_api.py (best_model.zip) and 
the final_model.zip is indeed 2-dimensional. However, some other models present in 
the models folder (ppo_pricing_profitable.zip and ppo_pricing.zip) have a 
3-dimensional observation space, which indicates they were trained with different 
environment definitions or additional state features. 
●  Actions:  The agent's decision at each step. Here, the action space is a continuous value 
representing the recommended_price. The price is clipped between 10.0 and 50.0. 
●  Rewards and Penalties:  The feedback mechanism guiding the agent's learning: 
○  Reward:  The total_profit accumulated from sales within an episode. 
○  Penalty:  If the episode terminates with leftover inventory, a penalty equal to the 
inventory multiplied by cost_per_item is subtracted from the final reward. This 
encourages the agent to sell all available inventory. 
3. 
�
�
 Tools, Libraries & Frameworks Used 
The project leverages a modern Python ecosystem for machine learning and web application 
development: 
●  Python:  The core programming language for the entire project. 
●  Streamlit:  Used for building the interactive web dashboard (dashboard.py), providing a 
simple and fast way to create data applications. 
●  FastAPI:  A high-performance web framework for building APIs with Python 3.7+, used for 
the prediction API (pricing_api.py). 
●  Gymnasium (formerly OpenAI Gym):  Provides the PricingEnvironment for simulating 
the pricing problem. The system_info.txt files confirm Gymnasium 0.29.1 and OpenAI Gym 
0.26.2. 
●  Stable-Baselines3:  A set of reliable implementations of reinforcement learning 
algorithms in PyTorch. The PPO algorithm is used from this library (train_model.py, 
pricing_api.py, b_testing.py, pricing_rl.py). All models indicate Stable-Baselines3 version 
2.1.0. 
●  Numpy:  Essential for numerical operations, especially for handling observation and 
action spaces within the environment and model (pricing_env.py, pricing_api.py, 
b_testing.py, pricing_rl.py). Version 1.26.4. 
●  Pandas:  Used in b_testing.py for data manipulation and generating summary statistics of 
evaluation results. 
●  Scikit-learn:  (Not explicitly imported in the provided .py files, but often part of ML 
projects for preprocessing or utility functions.) 
●  Requests:  Used by the Streamlit dashboard (dashboard.py) to make HTTP requests to 
the FastAPI backend. 
●  Uvicorn:  (Implicitly used to serve the FastAPI application). 
●  Matplotlib & Seaborn:  For creating visualizations in b_testing.py, such as profit over 
episodes and profit distribution. 
4. 
⚙
 Project Architecture & Structure 
The project follows a modular and well-structured approach, dividing functionalities into 
distinct files and folders: 
New folder/ 
├── b_testing.py 
├── dashboard.py 
├── pricing_api.py 
├── pricing_env.py 
├── pricing_pro.py.code-workspace 
├── pricing_rl.py 
├── phase1_test.png 
├── models/ 
│   ├── best_model.zip 
│   ├── final_model.zip 
│   ├── ppo_pricing.zip 
│   └── ... (other model-related files like pytorch_variables.pth, system_info.txt, 
_stable_baselines3_version) 
├── logs/ 
│   └── evaluations.npz 
├── eval_logs/ 
│   └── evaluations.npz 
├── ppo_logs/ 
│   └── PPO_1/ 
│       └── events.out.tfevents... 
└── __pycache__/ 
Files and their purpose: 
●  pricing_env.py  : This file defines the PricingEnvironment class, which is the core 
simulation of the dynamic pricing problem. It sets up the observation space, action 
space, and the logic for reset and step functions, including reward calculation and 
termination conditions. 
●  train_model.py  : This script is responsible for training the PPO agent. It initializes the 
PricingEnvironment, sets up PPO with specific hyperparameters, trains the model for a 
defined number of timesteps, and saves the best performing model (best_model.zip) and 
the final model (final_model.zip). 
●  pricing_api.py  : This file implements the FastAPI application. It loads the trained 
best_model.zip and exposes a /predict endpoint. This endpoint receives current state 
information (step, inventory, demand factor) from the dashboard, processes it, and 
returns a recommended price. 
●  dashboard.py  : This Streamlit script creates the interactive user interface. It allows users 
to input various environmental parameters (day, inventory, demand factor) using sliders 
and displays the price recommendation received from the pricing_api.py. 
●  b_testing.py  : This script is used for comprehensive evaluation, comparing the PPO 
agent's performance against baseline strategies (Fixed, Random, Rule-Based). It runs 
multiple episodes for each strategy and calculates various metrics like profit, average 
price, and price volatility. 
●  pricing_rl.py  : This file contains an alternative training script for the PPO model and also 
includes functions to test fixed pricing strategies and predict prices using a trained 
model. 
How components interact: 
1.  Training Pipeline:  train_model.py (or pricing_rl.py) uses pricing_env.py to create a 
simulated environment. The PPO algorithm from Stable-Baselines3 is trained within this 
environment. The trained models are saved as .zip files in the models/ directory. 
2.  API Service:  pricing_api.py loads one of the saved models (specifically 
models/best_model.zip in this case). It then runs as a web service, awaiting requests for 
price predictions. 
3.  User Interface:  dashboard.py provides the graphical interface for the user. When the 
user inputs parameters and requests a price, it sends an HTTP POST request to the 
/predict endpoint of the FastAPI application (pricing_api.py). 
4.  Prediction Flow:  The FastAPI app receives the input, passes it to the loaded PPO model 
for inference, and returns the recommended_price back to the dashboard. 
5.  Evaluation:  b_testing.py loads a pre-trained PPO model (ppo_pricing.zip) and evaluates 
its performance by interacting with instances of PricingEnvironment. 
5. 
�
�
 Implementation Phases 
 Phase 1: Environment Setup 
 The heart of any RL project is a well-designed environment. 
 The PricingEnvironment class in pricing_env.py defines the interaction space for the agent. 
 Python 
 import  numpy  as  np 
 import  gymnasium  as  gym 
 from  gymnasium  import  spaces 
 class  PricingEnvironment(gym.Env): 
 def  __init__  (self, max_inventory=  500  , max_steps=  30  , cost_per_item=  8  ): 
 super  (PricingEnvironment, self).__init__() 
 self.max_inventory = max_inventory 
 self.max_steps = max_steps 
 self.cost_per_item = cost_per_item 
 self.observation_space = spaces.Box( 
 low=np.array([  0.0  ,  0.0  ], dtype=np.float32), 
 high=np.array([  1.0  ,  1.0  ], dtype=np.float32), 
 dtype=np.float32, 
 ) 
 self.action_space = spaces.Box( 
 low=np.array([  10.0  ]), high=np.array([  50.0  ]), dtype=np.float32 
 ) 
 self.reset() 
 def  reset  (self, seed=  None  , options=  None  ): 
 super  ().reset(seed=seed) 
 self.current_step =  0 
 self.inventory = self.max_inventory 
 self.demand_factor = np.random.uniform(  0.7  ,  1.3  ) 
 self.total_profit =  0 
 return  self._get_obs(), {} 
 def  step  (self, action): 
 price =  float  (np.clip(action[  0  ]  if  isinstance  (action, (np.ndarray,  list  ,  tuple  ))  else  action,  10.0  , 
 50.0  )) 
 base_demand =  max  (  0  ,  100  - price) 
 demand =  int  (base_demand * self.demand_factor) 
 units_sold =  min  (demand, self.inventory) 
 self.inventory -= units_sold 
 self.current_step +=  1 
 self.total_profit += (price - self.cost_per_item) * units_sold 
 terminated = self.current_step >= self.max_steps  or  self.inventory <=  0 
 if  terminated: 
 reward = self.total_profit 
 if  self.inventory >  0  : 
 reward -= self.inventory * self.cost_per_item  # Penalty for leftover inventory 
 else  : 
 reward =  0  # No reward until the episode is over 
 return  self._get_obs(), reward, terminated,  False  , {} 
 def  _get_obs  (self): 
 normalized_inventory = self.inventory / self.max_inventory 
 normalized_demand = (self.demand_factor -  0.7  ) / (  1.3  -  0.7  ) 
 return  np.array([normalized_inventory, normalized_demand], dtype=np.float32) 
 def  render  (self): 
 print(  f"Step:  {self.current_step}  , Inventory:  {self.inventory}  , Demand Factor:  {self.demand_factor}  "  ) 
 ●  Designing the PricingEnvironment  : The environment simulates a discrete-time pricing 
 problem. It tracks current_step, inventory, demand_factor, and total_profit. 
 ●  State/Observation Space  : Defined as a Box space containing normalized_inventory and 
 normalized_demand. The normalization ensures that the input to the neural network is 
 within a consistent range, aiding stable training. 
 ●  Action Space  : Also a Box space, representing the single continuous action of setting the 
 price between 10.0 and 50.0. 
 ●  Reward Logic  : The primary reward is the total_profit accumulated over an episode. A 
 crucial enhancement is the inclusion of a penalty for leftover inventory at the end of an 
 episode. This incentivizes the agent to sell all its stock, aligning with business goals 
 beyond just high prices. 
 Phase 2: Model Training 
 The train_model.py script handles the training of the PPO agent. 
 Python 
 # From New folder/train_model.py 
 from  stable_baselines3  import  PPO 
 from  stable_baselines3.common.env_checker  import  check_env 
 from  stable_baselines3.common.callbacks  import  EvalCallback 
 from  pricing_env  import  PricingEnvironment 
 env = PricingEnvironment() 
 check_env(env) 
 eval_callback = EvalCallback(env, best_model_save_path=  './models/'  , 
 log_path=  './logs/'  , eval_freq=  500  , 
 deterministic=  True  , render=  False  ) 
 model = PPO( 
 "MlpPolicy"  , 
 env, 
 verbose=  1  , 
 n_steps=  2048  , 
 batch_size=  64  , 
 gamma=  0.999  ,  # Prioritize long-term rewards even more 
 gae_lambda=  0.95  , 
 ent_coef=  0.01  ,  # Encourage exploration 
 learning_rate=  0.0003  , 
 clip_range=  0.2 
 ) 
 model.learn(total_timesteps=  500_000  , callback=eval_callback) 
 model.save(  "models/final_model.zip"  ) 
 print(  "
 ✅
 Model training complete. The best model is saved as best_model.zip."  ) 
●  Using PPO from Stable-Baselines3  : The PPO algorithm is initialized with an MlpPolicy, 
meaning a multi-layer perceptron (MLP) neural network is used for both the actor (policy) 
and critic (value) functions. 
●  Hyperparameters used  : 
○  n_steps=2048: Number of steps to run in each environment per update. 
○  batch_size=64: Mini-batch size for policy and value updates. 
○  gamma=0.999: Discount factor, prioritizing long-term rewards. 
○  gae_lambda=0.95: Factor for Generalized Advantage Estimation (GAE). 
○  ent_coef=0.01: Entropy coefficient, encouraging exploration. 
○  learning_rate=0.0003: Adam optimizer learning rate. 
○  clip_range=0.2: Clipping parameter for PPO's clipped surrogate objective. 
●  Training pipeline and saving model  : The model is trained for 500,000 timesteps. An 
EvalCallback is used to periodically evaluate the model and save the best_model.zip 
based on performance, while the final_model.zip is saved at the end of training. Log files 
(e.g., evaluations.npz) are generated in the logs/ directory. 
Phase 3: API Development 
The pricing_api.py file sets up a FastAPI application to serve price recommendations. 
Python 
# From New folder/pricing_api.py 
from  fastapi  import  FastAPI 
from  fastapi.middleware.cors  import  CORSMiddleware 
from  pydantic  import  BaseModel 
import  numpy  as  np 
from  stable_baselines3  import  PPO 
app = FastAPI() 
app.add_middleware( 
CORSMiddleware, 
allow_origins=[  "*"  ], 
allow_credentials=  True  , 
allow_methods=[  "*"  ], 
allow_headers=[  "*"  ], 
) 
model = PPO.load(  "models/best_model.zip"  ) 
class  PriceRequest(BaseModel): 
step:  int 
inventory:  int 
demand_factor:  float 
@app.post(  "/predict"  ) 
def  predict_price  (data: PriceRequest): 
normalized_inventory = data.inventory /  500.0 
normalized_demand = (data.demand_factor -  0.7  ) / (  1.3  -  0.7  ) 
recommended_price =  float  (action[  0  ]) 
obs = np.array([normalized_inventory, normalized_demand], dtype=np.float32).reshape(  1  , -  1  ) 
action, _ = model.predict(obs, deterministic=  True  ) 
print(  f"[DEBUG] Input:  {obs}  , Action:  {action}  , Price:  {recommended_price}  "  ) 
return  {  "recommended_price"  : recommended_price} 
●  FastAPI setup  : A FastAPI instance is created with CORS (Cross-Origin Resource Sharing) 
middleware enabled to allow requests from any origin, which is useful for local 
development with the Streamlit dashboard. 
●  /predict endpoint structure and working  : The PPO model saved as best_model.zip is 
loaded at API startup. The /predict endpoint expects a JSON payload matching the 
PriceRequest Pydantic model (step, inventory, demand_factor). Inside the endpoint, the 
raw inputs are normalized to match the observation space of the trained model. The 
model then predicts an action (the optimal price) in a deterministic manner. The 
recommended price is returned as a JSON response. 
Phase 4: Interactive Dashboard 
The dashboard.py file builds a user-friendly interface using Streamlit. 
Python 
# From New folder/dashboard.py 
import  streamlit  as  st 
import  requests 
st.title(  "
 🧠
 Dynamic Pricing Dashboard"  ) 
step = st.slider(  "Current Day (Step)"  , min_value=  0  , max_value=  30  , value=  0  ) 
inventory = st.slider(  "Inventory Level"  , min_value=  0  , max_value=  500  , value=  250  ) 
demand_factor = st.slider(  "Demand Factor"  , min_value=  0.5  , max_value=  1.5  , value=  1.0  , 
step=  0.01  ) 
if  st.button(  "
 📈
 Get Price Recommendation"  ): 
try  : 
response = requests.post( 
"http://127.0.0.1:8000/predict"  , 
json={  "step"  : step,  "inventory"  : inventory,  "demand_factor"  : demand_factor}, 
timeout=  5 
) 
response.raise_for_status() 
result = response.json() 
st.success(  f"
 💰
 Recommended Price: **$  {result[  'recommended_price'  ]}  **"  ) 
except  requests.exceptions.RequestException  as  e: 
st.error(  f"
 ❌
 Prediction error:  {e}  "  ) 
●  Built using Streamlit  : The dashboard provides interactive sliders for Current Day, 
Inventory Level, and Demand Factor. 
●  Display of model-predicted price  : When the "Get Price Recommendation" button is 
clicked, the dashboard sends the current slider values to the FastAPI /predict endpoint. It 
then displays the received recommended_price or an error message if the API call fails. 
6. 
�
�
 UI/UX Design 
The dashboard prioritizes a clean and professional user interface, leveraging Streamlit's 
intuitive design capabilities. 
●  Simplicity  : The design focuses on essential input parameters (Current Day, Inventory 
Level, Demand Factor) presented via easy-to-use sliders. 
●  Clarity  : Clear titles and labels guide the user. The recommendation is displayed 
prominently with a success message. 
●  Visual Cues  : Emojis (
 🧠
 , 
�
�
 , 
�
�
 , 
❌
 ) are incorporated to enhance visual appeal and 
convey information concisely. 
●  Interactivity  : The sliders provide immediate feedback as values are adjusted, and the 
button clearly triggers the price recommendation process. The overall experience is 
designed to be straightforward for users with basic knowledge of dynamic pricing 
concepts. 
7. 
�
�
 Testing & Results 
The b_testing.py script provides a framework for evaluating the trained PPO model against 
various baseline strategies. 
●  Comparison Strategies  : The script compares PPO with: 
○  Fixed Price  : A constant price (e.g., $30). 
○  Random  : Prices are chosen randomly within the action space. 
○  Rule-Based  : A simple rule (e.g., higher price early, discounted price later). 
●  Metrics  : For each strategy, over multiple episodes (e.g., 20 episodes), the following 
metrics are calculated: 
○  Profit 
○  AvgPrice 
○  PriceVolatility 
○  InventoryUsed 
●  Observations and Learning Patterns  : The evaluation summary and plots generated by 
b_testing.py (such as "Profit per Episode by Strategy" and "Profit Distribution") would 
reveal how well the PPO agent performs compared to the baselines. The phase1_test.png 
image illustrates "Rewards Over Time" for a particular training run, showing an initial 
increase in profit around day 5, followed by a sharp decline. This indicates that model 
behavior changes across different time steps, and that rewards are not always 
monotonically increasing. Detailed analysis of these plots and summary statistics helps 
understand the PPO agent's learning effectiveness and its ability to adapt pricing to 
maximize profit. 
8. 
�
�
 Deployment Guide 
To set up and run the Dynamic Pricing project locally, follow these steps: 
1.  Clone the Repository (or create the folder structure): 
Ensure you have the entire project structure as described in Section 4. 
2.  Create a Python Virtual Environment (recommended using Conda): 
Bash 
conda create -n dynamic_pricing python=3.9 
conda activate dynamic_pricing 
3.  Install Dependencies: 
While a requirements.txt is not provided, the key libraries needed are: 
Bash 
pip install numpy gymnasium stable-baselines3[extra] fastapi uvicorn[standard] streamlit 
requests pandas matplotlib seaborn 
4.  Train the Model (Optional, if you want to retrain): 
If you wish to retrain the PPO model, navigate to the New folder directory and run: 
Bash 
python train_model.py 
This will save best_model.zip and final_model.zip in the models/ directory. You can also 
run python pricing_rl.py to train another model and test fixed pricing strategies. 
5.  Launch the FastAPI Backend: 
Navigate to the New folder directory and run the FastAPI application using Uvicorn: 
Bash 
uvicorn pricing_api:app --reload 
The --reload flag enables auto-reloading upon code changes, useful for development. 
The API will typically run on http://127.0.0.1:8000. 
6.  Launch the Streamlit Dashboard: 
Open a new terminal, navigate to the New folder directory, and run the Streamlit 
application: 
Bash 
streamlit run dashboard.py 
This will open the interactive dashboard in your web browser. 
9. 
�
�
 Challenges & Learnings 
Developing this reinforcement learning project involved several common challenges and key 
learnings: 
●  Observation Space Consistency  : One notable challenge was managing the observation 
space. The PricingEnvironment in pricing_env.py defines a 2-element observation 
(normalized inventory and normalized demand factor). The pricing_api.py correctly 
normalizes and uses a 2-element observation for prediction. However, the pricing_rl.py 
script attempts to use a 3-element observation including step ([step, inventory, 
demand_factor]), which would lead to a shape mismatch with models trained on the 
2-element space. Ensuring strict consistency between the environment's _get_obs 
method and how observations are constructed for model prediction is crucial. 
●  Reward Tuning and Shaping  : Designing an effective reward function was iterative. The 
initial profit accumulation was straightforward, but the addition of a penalty for leftover 
inventory was vital for encouraging the agent to deplete its stock, even if it meant 
adjusting prices downwards. This "reward shaping" helps guide the agent towards more 
desirable business outcomes. 
●  Gym vs. Gymnasium Transition  : The environment was developed using Gymnasium 
(gymnasium as gym), the actively maintained successor to OpenAI Gym. This required 
understanding and adapting to the slight API differences, such as the reset and step 
method signatures (obs, info from reset; obs, reward, terminated, truncated, info from 
step). The system information confirming Gymnasium 0.29.1 is key here. 
●  Robust Input Handling  : In pricing_env.py, the step method includes a check 
isinstance(action, (np.ndarray, list, tuple)) to handle various input formats for the action, 
making the environment more robust to different ways an action might be passed (e.g., a 
single float vs. a NumPy array). This small detail can prevent runtime errors during 
training or evaluation. 
●  Model Saving and Loading  : Stable-Baselines3's .save() and .load() methods for models 
are convenient, but it's important to remember that they save the entire model (policy, 
optimizer, environment parameters, etc.) within a .zip file. Ensuring the correct model 
(best_model.zip) is loaded by the API is critical for deploying the best-performing agent. 
10. 
�
�
 Future Improvements 
This project lays a solid foundation for dynamic pricing using RL, but there are many avenues 
for future enhancements: 
●  Multi-Agent Pricing  : Introduce competing agents that also adjust prices, simulating a 
more realistic market environment. This would explore game theory concepts in RL. 
●  Time-Based Demand Fluctuations  : Implement more sophisticated demand models 
where the demand_factor varies based on the time of day, week, or season, forcing the 
agent to adapt to non-uniform demand patterns. 
●  Inventory Cost Modeling  : Incorporate holding costs for unsold inventory and stockout 
costs for unmet demand. This would make the reward function more aligned with 
real-world supply chain economics. 
●  Real-world Dataset Integration  : Instead of a purely simulated base_demand, integrate 
historical sales data or real-time market data to train and evaluate the pricing agent. This 
would require data preprocessing and feature engineering. 
●  More Complex Network Architectures  : Experiment with deeper or wider neural 
networks for the PPO MlpPolicy, or explore other policy architectures if the 
observation/action spaces become more complex. 
●  Advanced Evaluation Metrics  : Beyond profit, evaluate metrics like customer 
satisfaction (e.g., from prices being too high or stockouts), market share, and long-term 
customer value. 
●  A/B Testing Integration  : For a real-world deployment, set up a framework for A/B 
testing different pricing strategies or agent versions to measure their impact directly on 
business KPIs. 
