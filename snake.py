class Snake:
	def __init__(self, positions, vel, head_color=RED, body_color=DARK_GREY, length=INITIAL_SNAKE_LENGTH):
		self.is_dead = False
		self.pressed_arrow_key = False
		self.pseudo_prediction = DO_NOTHING

		self.head_color = head_color
		self.body_color = body_color

		self.length = length

		if np.array_equal(vel, RANDOM_VEL):
			self.vel = random.choice([LEFT, RIGHT, UP, DOWN])
		else:
			self.vel = vel

		if np.array_equal(positions, RANDOM_POSITION):
			wrong_position = True
			while wrong_position:
				self.positions = [pick_random_position()]
				if self.not_going_into_wall(self.vel * self.length) and \
						self.not_going_into_wall(- self.vel * self.length):
					wrong_position = False
		else:
			self.positions = list(positions)

		self.bites = []
		self.new_potential_position = np.zeros(VECTOR_SIZE, int)
		self.new_potential_vel = np.zeros(VECTOR_SIZE)

		self.body_surface = pg.Surface(GRID_SIZE, depth=8)

		# Layers: own body, other bodies, heads, food, borders
		self.pers_surfaces = []
		self.states = []
		for i in range(LAYERS_PER_STATE):
			self.pers_surfaces.append(pg.Surface(GRID_SIZE, depth=8))
			self.states.append([])

		self.actions = []
		self.rewards = []
		self.time = 0

	def move(self, snakes):
		# not long enough:
		while len(self.positions) < self.length:
			self.positions.insert(0, self.positions[0] + self.get_tail_direction())
		# moving:
		if self.not_going_into_wall(self.vel) and self.not_going_into_snake_head(snakes):
			self.positions.append(self.new_potential_position)
		else:
			self.rewards[self.time] += RW_DID_NOT_MOVE
		# too long:
		while len(self.positions) > self.length:
			self.positions.pop(0)
		# dead:
		if len(self.positions) <= DEAD_AT_LENGTH:
			self.is_dead = True

	def check_has_eaten(self, food):
		for j in range(len(food)):
			if np.array_equal(self.positions[-1], food[j].position):
				self.rewards[self.time] += RW_HAS_EATEN
				self.length += 1
				food.pop(j)
				break

	def check_got_bitten(self, snakes):
		self.bites = [[], []]
		for snake in snakes:
			snake_head = snake.positions[-1].tolist()
			for potential_bite in range(len(self.positions)):
				if self.positions[potential_bite].tolist() == snake_head:  # If a snake bit me:
					if potential_bite < len(self.positions) - 1 \
							or potential_bite == len(self.positions) and snake != self:
						self.bites[0].append(potential_bite)
						self.bites[1].append(snake)
						if snake != self:
							self.rewards[self.time] += RW_GOT_BITTEN
							snake.rewards[snake.time] += RW_HAS_BITTEN
							snake.length += 1
						else:
							self.rewards[self.time] += RW_HAS_BITTEN_HIMSELF

		if len(self.bites[0]) > 0:
			new_snakes_positions = [[], []]

			# Separating self into two potential snakes:
			for j in range(len(self.positions)):
				if j < min(self.bites[0]):
					new_snakes_positions[0].append(self.positions[j])
				if j > max(self.bites[0]):
					new_snakes_positions[1].append(self.positions[j])

			# Reversing potential tail-snake:
			new_snakes_positions[0].reverse()
			new_snake_vel = [self.get_tail_direction(), self.vel]

			# Choosing the longest part for self:
			if len(new_snakes_positions[0]) > len(new_snakes_positions[1]):
				this_snake = 0
			else:
				this_snake = 1

			# Cutting self:
			self.length = len(new_snakes_positions[this_snake])
			self.positions = new_snakes_positions[this_snake]
			self.vel = new_snake_vel[this_snake]

			# Rewarding for kill:
			if self.length <= DEAD_AT_LENGTH:
				for snake in self.bites[1]:
					snake.rewards[snake.time] += RW_HAS_KILLED
				self.rewards[self.time] += RW_GOT_KILLED

			# Birthing new snake:
			new_len = len(new_snakes_positions[not this_snake])
			if new_len > BIRTH_FROM_LENGTH:
				snakes.append(Snake(new_snakes_positions[not this_snake],
									new_snake_vel[not this_snake],
									length=new_len))
				snakes[-1].rewards.append(RW_TIME_PASSED)

	def get_tail_direction(self):
		if len(self.positions) <= 1:
			return - self.vel
		else:
			return self.positions[0] - self.positions[1]

	def not_going_into_wall(self, direction):
		self.new_potential_position = self.positions[-1] + direction
		return np.all(np.logical_and(np.less_equal(np.zeros(VECTOR_SIZE), self.new_potential_position),
									 np.less(self.new_potential_position, GRID_SIZE)))

	def not_going_into_snake_head(self, snakes):
		good = True
		for snake in snakes:
			if snake.positions[-1].tolist() == self.new_potential_position.tolist():
				good = False
		return good

	def draw(self, screen, heads_surface):
		for cell_position in self.positions[:-1]:
			if not AI_TRAINING_MODE:
				fill_cell(screen, cell_position, self.body_color)
			self.body_surface.set_at(cell_position, WHITE)
		if not AI_TRAINING_MODE:
			fill_cell(screen, self.positions[-1], self.head_color)
		heads_surface.set_at(self.positions[-1], WHITE)
