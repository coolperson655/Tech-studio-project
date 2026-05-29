# import pygame
# CALIBRATION_RED = pygame.Color(252, 65, 3)
# CALIBRATION_YELLOW = pygame.Color(252, 219, 3)
# check_color_dict = {'IM_COLOR':['Waiting...',CALIBRATION_RED], 'RP_COLOR': ['Waiting...',CALIBRATION_RED], 'T_COLOR': ['Waiting...',CALIBRATION_RED]}
# print(check_color_dict["IM_COLOR"][1], check_color_dict["RP_COLOR"][1], check_color_dict["T_COLOR"][1])
# pygame.init()
# clock = pygame.time.Clock()
# while True:
#     pygame.display.set_caption("Calibration Test")

#     screen = pygame.display.set_mode((900, 600))
#     screen.fill(CALIBRATION_RED)
#     # pygame.draw.rect(screen, check_color_dict["IM_COLOR"][1], (40, 30, 440, 580), border_radius=10)
#     clock.tick(60)
testlist = [1,2,3]
print(testlist[0])
testlist[0] = 5
print(testlist[0])